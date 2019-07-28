# Основной модуль DolboNet
# by Wokashi RG
# https://github.com/wokashi-rg

import discord
from core.checking_client import CheckingClient
import config
from utils.tprint import log
import asyncio

log('Проверяю Discord токен...')
checking_client = CheckingClient()
login_successful = False
try:
    checking_client.run(config.token)
    login_successful = True
except discord.errors.LoginFailure:
    log('НЕВЕРНЫЙ DISCORD ТОКЕН! Необходимо отредактировать файл config.py!')

if login_successful:
    log('Discord токен проверен.')
    asyncio.set_event_loop(asyncio.new_event_loop())
    from core.main_client import MainClient
    main_client = MainClient()
    main_client.run(config.token)
