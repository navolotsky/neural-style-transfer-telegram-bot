import os
from urllib.parse import urljoin

from aiogram import executor

from .bot import BOT_API_TOKEN, dp, on_shutdown, on_startup, bot


WEBHOOK_HOST_ADDR = os.environ['WEBHOOK_HOST_ADDR']
WEBHOOK_PATH = f"/webhook/{BOT_API_TOKEN}"
WEBHOOK_URL = urljoin(WEBHOOK_HOST_ADDR, WEBHOOK_PATH)

WEBAPP_HOST = os.environ.get("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = os.environ['PORT']


async def on_startup_webhook(dp, webhook_url=WEBHOOK_URL):
    await on_startup(dp)
    await bot.set_webhook(webhook_url)

if __name__ == '__main__':
    executor.start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup_webhook,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
