# Тестовый Discord клиент для проверки валидности Discord токена
# by Wokashi RG
# https://github.com/wokashi-rg

import discord


class CheckingClient(discord.Client):

    async def on_ready(self):
        await self.logout()
