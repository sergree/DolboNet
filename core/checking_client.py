# Тестовый Discord клиент для проверки валидности Discord токена
# by Sergree
# https://github.com/sergree

import discord


class CheckingClient(discord.Client):
    def __init__(self, **options):
        super().__init__(**options)

    async def on_ready(self):
        await self.logout()
