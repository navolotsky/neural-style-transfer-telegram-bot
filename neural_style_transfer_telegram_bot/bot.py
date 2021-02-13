import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urljoin

from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types.input_file import InputFile
from aiogram.types.input_media import InputMediaDocument, InputMediaPhoto

from .inference import (ImageProcessingError, ImageTooBigError,
                        ImageTooSmallError, Task, TaskType, make_inference)

BOT_API_TOKEN = os.environ['BOT_API_TOKEN']

WEBHOOK_HOST_ADDR = os.environ['WEBHOOK_HOST_ADDR']
WEBHOOK_PATH = f"/webhook/{BOT_API_TOKEN}"
WEBHOOK_URL = urljoin(WEBHOOK_HOST_ADDR, WEBHOOK_PATH)

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = os.environ['PORT']


LIB_LOGGING_LEVEL = logging.INFO
APP_LOGGING_LEVEL = logging.DEBUG

# Due to length limits (>=2, <=10) of "media" array
# in "sendMediaGroup" Telegram API method:
MAX_PICS_PER_REQUEST = 10

logging.basicConfig(level=LIB_LOGGING_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(APP_LOGGING_LEVEL)

bot = Bot(token=BOT_API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
pool = ProcessPoolExecutor(max_workers=1)


task_types_to_buttons = {
    TaskType.style_transfer: "Style transfer to one image from another",
    TaskType.photo2van_gogh: "Artist: Vincent Willem van Gogh",
    TaskType.photo2monet: "Artist: Oscar-Claude Monet",
    TaskType.photo2cezanne: "Artist: Paul Cézann",
    TaskType.photo2ukiyoe: "Genre: Ukiyo-e"
}
choosing_task_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
choosing_task_keyboard.add(task_types_to_buttons[TaskType.style_transfer])
choosing_task_keyboard.row(
    task_types_to_buttons[TaskType.photo2van_gogh],
    task_types_to_buttons[TaskType.photo2monet],
    task_types_to_buttons[TaskType.photo2cezanne])
choosing_task_keyboard.add(task_types_to_buttons[TaskType.photo2ukiyoe])


def get_max_pics_per_request(task_type):
    max_pics_per_request = MAX_PICS_PER_REQUEST
    if task_type is TaskType.style_transfer:
        max_pics_per_request -= max_pics_per_request % 2
    return max_pics_per_request


class StylizationRequest(StatesGroup):
    waiting_for_style_chosen = State()
    waiting_for_images = State()
    processing = State()


REQUESTS_IN_PROCESSING = {}


@dp.message_handler(commands=['start', 'help', 'about', 'commands', 'request'])
async def send_welcome(message: types.Message):
    await StylizationRequest.waiting_for_style_chosen.set()
    if message.get_command(pure=True) != 'request':
        answer_text = "Hi! I'm a bot for neural style transfer. "
    else:
        answer_text = ""
    await message.answer(
        answer_text +
        "Please choose what you want to do:"
        "\n\t— style transfer to one image from another you'll give"
        "\n\t— or a one of the provided styles that you want "
        "to transfer to your images.", reply_markup=choosing_task_keyboard)


@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(contains='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.reply(
            "Nothing to cancel. /request to start a new request.",
            reply_markup=types.ReplyKeyboardRemove())
        return
    logger.debug("Cancelling state %r", current_state)
    await state.finish()
    future = REQUESTS_IN_PROCESSING.pop(message.from_user.id, None)
    if future is not None:
        future.cancel()
        logger.debug("Future cancelling requested")
        try:
            await future
        except asyncio.CancelledError:
            logger.debug("Future cancelled")
        except ImageProcessingError as exc:
            # Task is cancelled so there is no need to process error.
            logger.debug(
                "got exception when awaiting cancelled inference",
                exc_info=exc)
        except Exception:
            logger.exception(
                "got unexpected exception when awaiting cancelled inference")
        if future.cancelled():
            await message.reply(
                "The request in processing cancelled. Start a new /request?",
                reply_markup=types.ReplyKeyboardRemove())
            return
        elif future.done():
            await message.reply(
                "The request already done. Start a new /request?",
                reply_markup=types.ReplyKeyboardRemove())
            return
    await message.reply(
        "Current scheduled request cancelled. Start a new /request?",
        reply_markup=types.ReplyKeyboardRemove())


async def style_chosen_handler(message: types.Message, state: FSMContext):
    for task_type, style in task_types_to_buttons.items():
        if style in message.text:
            break
    else:
        raise RuntimeError
    await state.update_data(task_type=task_type, images=[])
    await StylizationRequest.next()
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add("OK (get results as photos, by default)")
    keyboard.add("OK (get results as files, less compression)")
    keyboard.add("Cancel")
    answer_text = (
        f"You chose: {style}.\n\nNow please send "
        f"up to {get_max_pics_per_request(task_type)} images. ")
    if task_type is TaskType.style_transfer:
        answer_text += (
            "Number of images must be even and images must be sent one by one "
            "as follows:\n\n\ttarget image #1\n\tstyle image #1"
            "\n\ttarget image #2\n\tstyle image #2\n\t...")
    answer_text += (
        "\n\nThen please wait until uploading finished "
        "(otherwise images can be missed) and "
        "press OK to start processing images or press Cancel at any time "
        "if you changed your mind and want to do something different.")
    await message.answer(answer_text, reply_markup=keyboard)

# Add several "contains" conditions for style_chosen_handler():
for value in task_types_to_buttons.values():
    dp.register_message_handler(
        style_chosen_handler,
        Text(contains=value),
        state=StylizationRequest.waiting_for_style_chosen)


@dp.message_handler(
    state=StylizationRequest.waiting_for_images,
    content_types=types.ContentTypes.PHOTO | types.ContentTypes.DOCUMENT)
async def save_images_as_mediagroup(message: types.Message, state: FSMContext):
    if message.content_type is types.ContentType.PHOTO:
        ext = '.jpg'
        image = message.photo[-1]
    elif message.content_type is types.ContentType.DOCUMENT:
        image = message.document
        if message.document.mime_type == 'image/jpeg':
            ext = '.jpg'
        elif message.document.mime_type == 'image/png':
            ext = '.png'
        else:
            raise ValueError(
                f"unsupported mime_type {message.document.mime_type}")
    else:
        raise TypeError("content_type must be PHOTO or DOCUMENT")
    async with state.proxy() as proxy:
        proxy['images'].append((image, ext))


@dp.message_handler(
    Text(contains='OK', ignore_case=True),
    state=StylizationRequest.waiting_for_images)
async def process_images(message: types.Message, state: FSMContext):
    if "get results as files" in message.text.lower():
        def GroupElement(path): return InputMediaDocument(InputFile(path))
    else:
        def GroupElement(path): return InputMediaPhoto(InputFile(path))

    data = await state.get_data()
    task_type = data['task_type']
    images = data['images']
    answer_text = ""
    if not images:
        await message.answer(
            "You sent no images. "
            "Please send at least one or /cancel the request.")
        return

    max_pics_per_request = get_max_pics_per_request(task_type)
    if len(images) > max_pics_per_request:
        answer_text = (f"Sorry, you sent too many images. "
                       f"Up to {max_pics_per_request} allowed.")
        above_limit = len(images) - max_pics_per_request
        if above_limit == 1:
            answer_text += f"{above_limit} image "
        else:
            answer_text += f"{above_limit} images "
        answer_text += "above the limit will be ignored.\n\n"
        del images[max_pics_per_request:]
    elif (
        task_type is TaskType.style_transfer and
        len(images) % 2 != 0
    ):
        if len(images) == 1:
            await message.answer(
                "You sent only one image. "
                "A second one is required as a style source. "
                "Please send more images or /cancel the request.")
            return
        answer_text = (f"Sorry, you sent odd number of images. "
                       f"Even number up to {max_pics_per_request} allowed. ")
        above_limit = len(images) % 2
        if above_limit == 1:
            answer_text += f"{above_limit} last image "
        else:
            answer_text += f"{above_limit} last images "
        answer_text += "will be ignored.\n\n"
        del images[len(images) - above_limit:]
    await message.answer(
        answer_text +
        "Please wait for your images to be processed or /cancel the request.",
        reply_markup=types.ReplyKeyboardRemove())
    await StylizationRequest.next()
    task = Task(task_type)
    folder_path = task.dataroot
    for i, (image, ext) in enumerate(images):
        path = os.path.join(folder_path, str(i) + ext)
        await image.download(path)
        if await state.get_state() != StylizationRequest.processing.state:
            logger.debug("The request superseded by another one "
                         "while downloading images.")
            try:
                task.done()
            except Exception as exc:
                logger.exception(
                    "got unexpected exception when cleaning up: case #1")
            return
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(pool, make_inference, task)
    cur_req = REQUESTS_IN_PROCESSING.setdefault(message.from_user.id, future)
    if cur_req is not future:
        logger.debug("The request superseded by other one.")
        # Potentially PermissionError because the subprocess is
        # keeping the directory until computation has ended
        # even after cancelling the future:
        try:
            task.done()
        except Exception as exc:
            logger.exception(
                "got unexpected exception when cleaning up: case #2")
        return
    try:
        await future
        logger.debug("Future is done")
    except asyncio.CancelledError:
        logger.debug("Future is cancelled by other handler")
    except ImageTooBigError as exc:
        logger.debug("got exception when awaiting inference", exc_info=exc)
        await state.finish()
        await message.answer(
            "Sorry, couldn't process your images: all or some of them are too "
            "big. Try another /request?")
    except ImageTooSmallError as exc:
        logger.debug("got exception when awaiting inference", exc_info=exc)
        await state.finish()
        await message.answer(
            "Sorry, couldn't process your images: all or some of them are too "
            "small. Try another /request?")
    except Exception:
        logger.exception("got unexpected exception when awaiting inference")
        await state.finish()
        await message.answer(
            "Sorry, couldn't process your images due to unknown error. "
            "Try another /request?")
    else:
        res_dir = task.results_dir
        images = [os.path.join(res_dir, name) for name in os.listdir(res_dir)]
        logger.debug("Number of processed images: %d", len(images))
        await state.finish()
        await message.answer_media_group(
            [GroupElement(image) for image in images],
            disable_notification=True)
        await message.answer("Here is the result! Start a new /request?")
    finally:
        if REQUESTS_IN_PROCESSING.get(message.from_user.id) is future:
            REQUESTS_IN_PROCESSING.pop(message.from_user.id, None)
        # the same issue as above:
        try:
            task.done()
        except Exception as exc:
            logger.exception(
                "got unexpected exception when cleaning up: case #3")


@dp.message_handler(state="*")
async def any_other_message_handler(message: types.Message, state: FSMContext):
    cur_state = await state.get_state()
    if cur_state is None:
        await message.reply(
            "The command is not recognized. Please see the /help",
            reply_markup=types.ReplyKeyboardRemove())
    else:
        await message.reply(
            "Incorrent message for the current request. "
            "Please /cancel the request or "
            "see previous messages to remember what is going on.")
    return


async def on_startup(dp):
    logging.warning('Starting...')
    await bot.set_webhook(WEBHOOK_URL)


async def on_shutdown(dp):
    logging.warning('Shutting down...')
    logging.warning('Bye!')

if __name__ == '__main__':
    executor.start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
