from aiogram import executor

from .bot import dp, on_shutdown, on_startup

if __name__ == '__main__':
    executor.start_polling(
        dp,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True)
