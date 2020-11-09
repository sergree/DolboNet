# Микромодуль для вывода информации на консоль с датой и временем
# by Sergree
# https://github.com/sergree

import datetime


def current_time():
    return str(datetime.datetime.now()) + ": "


def log(*msg):
    print(current_time() + " ".join([str(x) for x in msg]))
