import sounddevice as sd
import numpy as np
import soundfile as sf
import whisper
import torch
from pynput import keyboard
import threading
import queue
import time
import tempfile
import os
import logging
import openai
import pyperclip
from playsound import playsound
from dotenv import load_dotenv
import customtkinter as ctk
import json
import pyautogui
from screeninfo import get_monitors

try:
    from pynput.mouse import Controller as MouseController

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

load_dotenv()

# --- Настройка ---
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")
HOTKEY = 'f9'
SETTINGS_HOTKEY = 'f10'
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_SIZE = "small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
GPT_MODEL = "gpt-4.1-mini"
# --- Новые настройки для отладки ---
SAVE_AUDIO_FOR_DEBUG = True
FORCE_FP32 = True
TARGET_DEVICE_INDEX = None

# --- Настройка логирования (Уровень INFO по умолчанию) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- Глобальные переменные ---
audio_queue = queue.Queue()
is_recording = False
stream = None
model = None
gpt_client = None
settings_window = None
prompt_configs = {}
processing_mode = "Коррекция"
recording_thread = None
processing_thread = None


# --- Функции ---

def load_prompts(file_path: str) -> dict:
    """Loads prompts from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            return prompts
    except FileNotFoundError:
        logging.error(f"Файл с промптами не найден: {file_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Ошибка декодирования JSON в файле: {file_path}")
        return {}


def play_sound_async(sound_name: str):
    """Воспроизводит звук в отдельном потоке-демоне."""
    sound_path = os.path.join(ASSETS_DIR, sound_name)

    def target():
        try:
            playsound(sound_path)
        except Exception as e:
            logging.warning(f"Не удалось воспроизвести звук '{sound_path}': {e}")

    # Поток-демон завершится, если основная программа остановится
    sound_thread = threading.Thread(target=target, daemon=True)
    sound_thread.start()


def process_text_with_gpt(text: str) -> str:
    """Processes the given text using the GPT model based on the current processing_mode."""
    if not gpt_client:
        logging.warning("Клиент OpenAI не инициализирован. Пропуск обработки GPT.")
        return text

    system_prompt = prompt_configs.get(processing_mode)
    if not system_prompt:
        logging.warning(f"Промпт для режима '{processing_mode}' не найден. Пропуск обработки GPT.")
        return text

    try:
        response = gpt_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=2048
        )

        processed_text = response.choices[0].message.content.strip()
        return processed_text

    except Exception as e:
        logging.error(f"Ошибка во время обработки текста через GPT: {e}", exc_info=True)
        return text


def paste_text(text: str):
    """Копирует текст в буфер и симулирует вставку (Ctrl+V или Cmd+V)."""
    try:
        pyperclip.copy(text)
        time.sleep(0.1)  # Пауза для буфера обмена

        logging.debug("Отправка команды вставки...")
        pyautogui.hotkey('ctrl', 'v')  # Для Windows/Linux
        # Примечание: для macOS pyautogui автоматически преобразует 'ctrl' в 'command'

        logging.debug("Команда вставки отправлена.")

    except Exception as e:
        logging.error(f"Ошибка при копировании/вставке: {e}", exc_info=True)


def audio_callback(indata, frames, time, status):
    """Callback для аудиоданных."""
    if not is_recording:
        return
    if status:
        # Оставляем предупреждение, т.к. это важно
        logging.warning(f"Статус аудиопотока: {status}")
    audio_queue.put(indata.copy())


def start_recording_thread_target():
    """Целевая функция для потока старта записи."""
    global stream, audio_queue, is_recording
    qsize = audio_queue.qsize()
    if qsize > 0:
        logging.debug(f"Очистка очереди перед записью (было {qsize} элементов)")
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

    logging.info("Начало записи...")  # <--- INFO: Ключевое событие
    try:
        logging.debug(
            f"Используем устройство: {'по умолчанию' if TARGET_DEVICE_INDEX is None else f'индекс {TARGET_DEVICE_INDEX}'}")
        stream = sd.InputStream(
            device=TARGET_DEVICE_INDEX,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            callback=audio_callback
        )
        stream.start()
        logging.debug("Аудиопоток запущен.")  # <--- DEBUG: Техническая деталь
    except Exception as e:
        logging.error(f"Ошибка при старте InputStream: {e}", exc_info=True)  # <--- ERROR: Важно
        is_recording = False
        if stream:
            try:
                stream.stop()
            except:
                pass
            try:
                stream.close()
            except:
                pass
        stream = None


def stop_recording_and_transcribe_thread_target():
    """Целевая функция для потока остановки и обработки."""
    global stream, audio_queue, model
    logging.info("Обработка записи...")

    active_stream = stream
    stream = None

    if active_stream:
        logging.debug("Остановка аудиопотока...")
        try:
            time.sleep(0.1)
            active_stream.stop()
            active_stream.close()
            logging.debug("Аудиопоток остановлен и закрыт.")
        except Exception as e:
            logging.error(f"Ошибка при остановке/закрытии потока: {e}", exc_info=True)
    else:
        logging.warning("Объект потока не найден при попытке остановки.")

    audio_data = []
    while not audio_queue.empty():
        try:
            audio_data.append(audio_queue.get_nowait())
        except queue.Empty:
            break

    if not audio_data:
        logging.warning("Нет аудио данных для обработки.")
        return

    try:
        full_audio_float32 = np.concatenate(audio_data, axis=0)
        full_audio_int16 = (full_audio_float32 * 32767).astype(np.int16)
    except ValueError as e:
        logging.error(f"Ошибка при объединении или конвертации аудио: {e}. Пропускаем.", exc_info=True)
        return

    final_path_to_use = None
    temp_file_context = None

    try:
        # --- ШАГ 1: Сохранение аудио в файл ---
        if SAVE_AUDIO_FOR_DEBUG:
            debug_audio_path = os.path.join(os.getcwd(), "last_recording_debug.wav")
            final_path_to_use = debug_audio_path
            logging.info(f"Сохранение отладочного аудио в: {final_path_to_use}")
            sf.write(final_path_to_use, full_audio_int16, SAMPLE_RATE, subtype='PCM_16')
        else:
            temp_file_context = tempfile.NamedTemporaryFile(suffix=".wav", delete=True, mode='wb')
            tmpfile = temp_file_context.__enter__()
            final_path_to_use = tmpfile.name
            logging.debug(f"Сохранение аудио во временный файл: {final_path_to_use}")
            sf.write(final_path_to_use, full_audio_int16, SAMPLE_RATE, subtype='PCM_16')
            tmpfile.flush()

        # --- ШАГ 2: ЕДИНЫЙ блок распознавания и обработки ---
        if final_path_to_use and model:
            logging.info("Распознавание речи...")
            start_time = time.time()
            use_fp16 = (DEVICE == 'cuda') and (not FORCE_FP32)
            logging.debug(f"Запуск Whisper с fp16={use_fp16}")
            result = model.transcribe(final_path_to_use, fp16=use_fp16)
            end_time = time.time()
            recognized_text = result["text"].strip()
            logging.info(f"Распознавание завершено за {end_time - start_time:.2f} сек.")

            if recognized_text:
                logging.info(f"Распознанный текст: '{recognized_text}'")
                logging.info(f"Обработка текста с помощью GPT в режиме '{processing_mode}'...")
                processed_text = process_text_with_gpt(recognized_text)
                logging.info(f"Обработанный текст: '{processed_text}'")
                paste_text(processed_text)
            else:
                logging.warning("Распознан пустой текст.")
                if SAVE_AUDIO_FOR_DEBUG:
                    logging.warning(f"Аудиозапись сохранена в '{final_path_to_use}' для проверки.")

        elif not model:
            logging.error("Модель Whisper не загружена!")

    except Exception as e:
        logging.exception(f"Непредвиденная ошибка в потоке обработки: {e}")
    finally:
        if temp_file_context:
            temp_file_context.__exit__(None, None, None)
        logging.debug("Поток обработки завершен.")


def reposition_window(window: ctk.CTk):
    """Calculates and applies the optimal window position based on monitor configuration."""
    WIN_WIDTH = 400
    WIN_HEIGHT = 350

    if PYNPUT_AVAILABLE:
        try:
            # 1. Получаем позицию мыши и список всех мониторов
            mouse_controller = MouseController()
            mouse_x, mouse_y = mouse_controller.position
            monitors = get_monitors()

            # 2. Находим, на каком мониторе сейчас курсор
            current_monitor = None
            for monitor in monitors:
                if monitor.x <= mouse_x < monitor.x + monitor.width and \
                   monitor.y <= mouse_y < monitor.y + monitor.height:
                    current_monitor = monitor
                    break

            # Если по какой-то причине не нашли, используем основной
            if current_monitor is None:
                current_monitor = next(m for m in monitors if m.is_primary)

            # 3. Рассчитываем позицию ОТНОСИТЕЛЬНО АКТИВНОГО МОНИТОРА
            # Логика по оси X
            if mouse_x + WIN_WIDTH > current_monitor.x + current_monitor.width:
                final_x = mouse_x - WIN_WIDTH - 20
            else:
                final_x = mouse_x + 20

            # Логика по оси Y
            if mouse_y + WIN_HEIGHT > current_monitor.y + current_monitor.height:
                final_y = mouse_y - WIN_HEIGHT - 20
            else:
                final_y = mouse_y + 20

            # 4. Применяем геометрию. Удаляем max(0, ...), так как X и Y могут быть отрицательными
            window.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{final_x}+{final_y}")

        except Exception as e:
            logging.warning(f"Ошибка при позиционировании окна: {e}. Окно будет центрировано.")
            # Центрируем окно, если что-то пошло не так
            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()
            center_x = (screen_width - WIN_WIDTH) // 2
            center_y = (screen_height - WIN_HEIGHT) // 2
            window.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{center_x}+{center_y}")
    else:
        # Fallback, если pynput недоступен
        window.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")


def show_settings_window():
    """Показывает существующее окно, обновляет его позицию и сбрасывает подсветку."""
    global settings_window, processing_mode
    if not settings_window:
        return

    reposition_window(settings_window) # Позиционируем, как и раньше

    # --- СБРОС И СИНХРОНИЗАЦИЯ СОСТОЯНИЯ ---
    # 1. Получаем список кнопок (он был сохранен при создании)
    buttons = settings_window.buttons
    mode_keys = settings_window.mode_keys

    # 2. Находим АКТУАЛЬНЫЙ индекс на основе "источника правды"
    try:
        current_index = mode_keys.index(processing_mode)
    except ValueError:
        current_index = 0

    # 3. Вызываем функцию подсветки, чтобы перерисовать окно
    settings_window.update_selection(current_index)
    # --- КОНЕЦ СИНХРОНИЗАЦИИ ---

    # Показываем и фокусируемся
    settings_window.deiconify()
    settings_window.lift()
    settings_window.focus_force()


def create_settings_window():
    """Создает и управляет окном настроек, реализуя навигацию и подсветку."""
    global settings_window, processing_mode

    # --- 1. Создание и базовые настройки окна ---
    window = ctk.CTk()
    window.title("Настройки")
    window.attributes("-topmost", True)
    settings_window = window  # Регистрируем окно глобально

    # --- 2. Локальные переменные и вложенные функции ---
    selected_index = 0  # Локальная переменная для временного состояния навигации
    buttons = []

    try:
        DEFAULT_COLOR = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        ACCENT_COLOR = ctk.ThemeManager.theme["CTkButton"]["hover_color"]
    except (KeyError, AttributeError):
        logging.warning("Не удалось получить цвета из темы. Используются цвета по умолчанию.")
        DEFAULT_COLOR = ("#3a7ebf", "#1f538d")
        ACCENT_COLOR = "#36719F"

    def update_selection(new_index):
        nonlocal selected_index
        if not buttons: return

        for i, btn in enumerate(buttons):
            if i == new_index:
                btn.configure(fg_color=ACCENT_COLOR)
            else:
                btn.configure(fg_color=DEFAULT_COLOR)

        selected_index = new_index
        buttons[selected_index].focus_set()

    def on_closing():
        logging.debug("Прячем окно настроек.")
        window.withdraw()  # Прячем, а не уничтожаем

    def select_mode(mode_name):
        global processing_mode
        processing_mode = mode_name
        logging.info(f"Режим изменен на '{mode_name}'.")
        on_closing()

    def handle_key_press(event):
        nonlocal selected_index
        if not buttons: return

        if event.keysym == 'Down':
            update_selection((selected_index + 1) % len(buttons))
        elif event.keysym == 'Up':
            update_selection((selected_index - 1 + len(buttons)) % len(buttons))
        elif event.keysym == 'Return':
            buttons[selected_index].invoke()
        elif event.keysym == 'Escape':
            on_closing()

    # --- 3. Привязка обработчиков и данных к объекту окна ---
    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.bind("<Up>", handle_key_press)
    window.bind("<Down>", handle_key_press)
    window.bind("<Return>", handle_key_press)
    window.bind("<Escape>", handle_key_press)

    # Прикрепляем важные данные к объекту окна для доступа извне
    window.buttons = buttons
    window.mode_keys = list(prompt_configs.keys())
    window.update_selection = update_selection

    # --- 4. Создание виджетов ---
    scrollable_frame = ctk.CTkScrollableFrame(window, fg_color="transparent")
    scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)

    for mode in window.mode_keys:
        btn = ctk.CTkButton(
            scrollable_frame,
            text=mode,
            command=lambda m=mode: select_mode(m)
        )
        btn.pack(fill="x", padx=5, pady=2)
        buttons.append(btn)

    # --- 5. Первичный показ окна ---
    # Позиционирование и подсветка теперь полностью управляются `show_settings_window`
    show_settings_window()

    window.mainloop()


def on_press(key):
    """Handles key press events from the listener."""
    try:
        if key == keyboard.Key.f9:
            on_f9_press()
        elif key == keyboard.Key.f10:
            on_f10_press()
    except Exception as e:
        logging.error(f"Unhandled exception in on_press: {e}", exc_info=True)


def on_release(key):
    """Handles key release events from the listener."""
    try:
        if key == keyboard.Key.f9:
            on_f9_release()
    except Exception as e:
        logging.error(f"Unhandled exception in on_release: {e}", exc_info=True)


# --- Функции-обработчики для hotkey ---
def on_f9_press():
    """Обработчик НАЖАТИЯ F9."""
    global is_recording, recording_thread
    logging.debug("Событие НАЖАТИЯ F9 обнаружено.")
    if not is_recording:
        play_sound_async('start.wav')
        is_recording = True
        recording_thread = threading.Thread(target=start_recording_thread_target, name="RecordingStartThread",
                                            daemon=True)
        recording_thread.start()
        # Добавляем минимальную задержку для предотвращения гонки состояний
        time.sleep(0.1)
    else:
        logging.debug("Нажатие F9: Запись уже идет, игнорируем.")


def on_f9_release():
    """Обработчик ОТПУСКАНИЯ F9."""
    global is_recording, processing_thread
    logging.debug("Событие ОТПУСКАНИЯ F9 обнаружено.")
    if is_recording:
        play_sound_async('stop.wav')
        is_recording = False
        processing_thread = threading.Thread(target=stop_recording_and_transcribe_thread_target,
                                             name="ProcessingThread", daemon=True)
        processing_thread.start()
    else:
        logging.debug("Отпускание F9: Запись не шла, игнорируем.")


def on_f10_press():
    """Handles F10 key press to open or show the settings window."""
    global settings_window

    if settings_window is None:
        # Если окно еще не создано, запускаем поток для его создания
        logging.debug("F10: Окно не создано, запускаем создание...")
        settings_thread = threading.Thread(target=create_settings_window, name="SettingsWindowThread", daemon=True)
        settings_thread.start()
    else:
        # Если окно уже существует (просто спрятано), показываем его
        logging.debug("F10: Окно существует, показываем его...")
        # Используем .after() для потокобезопасного вызова
        settings_window.after(0, show_settings_window)


# --- Основная часть ---
if __name__ == "__main__":
    prompt_configs = load_prompts(PROMPT_FILE)
    if not prompt_configs:
        logging.warning("Промпты не были загружены. Функционал GPT будет ограничен.")
    else:
        # Устанавливаем режим по умолчанию на первый ключ из конфига, если он есть
        processing_mode = next(iter(prompt_configs))
        logging.info(f"Загружено {len(prompt_configs)} режимов обработки. Режим по умолчанию: '{processing_mode}'")

    # Блок вывода устройств можно оставить INFO или закомментировать, т.к. он только при старте
    try:
        logging.info("Инициализация: Доступные аудиоустройства:")
        print("--------------------")
        print(sd.query_devices())
        print("--------------------")
        # ... (остальная логика определения устройства) ...
    except Exception as e:
        logging.error(f"Инициализация: Ошибка при запросе аудиоустройств: {e}")

    # Инициализация клиента OpenAI, если есть ключи
    if OPENAI_API_KEY and OPENAI_API_BASE:
        logging.info("Инициализация: Настройка клиента OpenAI...")
        try:
            gpt_client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE,
            )
            logging.info("Инициализация: Клиент OpenAI успешно настроен.")
        except Exception as e:
            logging.error(f"Инициализация: Не удалось настроить клиент OpenAI: {e}")
    else:
        logging.warning("Инициализация: Ключи OpenAI не найдены в .env, коррекция GPT будет пропущена.")

    logging.info(f"Инициализация: Проверка доступности GPU... Устройство для вычислений: {DEVICE.upper()}")
    logging.info("Инициализация: Загрузка модели Whisper...")
    try:
        model = whisper.load_model(MODEL_SIZE, device=DEVICE)
        logging.info(f"Инициализация: Модель '{MODEL_SIZE}' загружена на устройство: {DEVICE}")

        # --- "ПРОГРЕВ" МОДЕЛИ ---
        logging.info("Инициализация: Выполнение 'прогрева' модели для ускорения первого запуска...")
        # Создаем 1 секунду тишины
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        # Распознаем тишину
        model.transcribe(dummy_audio, fp16=(DEVICE == 'cuda' and not FORCE_FP32))
        logging.info("Инициализация: Модель 'прогрета' и готова к работе.")
        # --- КОНЕЦ "ПРОГРЕВА" ---
    except Exception as e:
        logging.error(f"Инициализация: Не удалось загрузить модель Whisper: {e}", exc_info=True)
        exit(1)

    logging.info("Инициализация: Запуск слушателя горячих клавиш...")
    # Listener is defined here and started. It will be stopped in the finally block.
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    logging.info(f"Инициализация: Слушатель горячих клавиш запущен. F9 - запись, F10 - настройки.")

    logging.info(f"=== Скрипт готов. F9 - запись голоса, F10 - настройки. ===")
    if SAVE_AUDIO_FOR_DEBUG or FORCE_FP32 or TARGET_DEVICE_INDEX is not None:
        logging.info(
            f"--- Активные отладочные настройки: SAVE_AUDIO={SAVE_AUDIO_FOR_DEBUG}, FORCE_FP32={FORCE_FP32}, DEVICE_INDEX={TARGET_DEVICE_INDEX} ---")  # <--- INFO (только если есть)
    logging.info("=== Чтобы остановить скрипт, нажмите Ctrl+C в консоли. ===")  # <--- INFO

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Получен сигнал KeyboardInterrupt. Завершение работы...")  # <--- INFO
    except Exception as e:
        logging.exception(f"Непредвиденная ошибка в главном цикле: {e}")  # <--- ERROR
    finally:
        logging.info("Завершение: Остановка слушателя клавиатуры...")
        listener.stop()
        if is_recording and stream:
            logging.warning("Завершение: Принудительная остановка аудиопотока...")  # <--- WARNING
            is_recording = False
            try:
                stream.stop()
                stream.close()
                logging.info("Завершение: Аудиопоток принудительно остановлен.")  # <--- INFO
            except Exception as e:
                logging.error(f"Завершение: Ошибка при принудительной остановке потока: {e}")  # <--- ERROR
        logging.info("=== Скрипт остановлен. ===")  # <--- INFO