# https://lowvoltage.github.io/2017/07/29/Yadisk-Direct-Download-Python

import requests

from utils.tprint import log

API_ENDPOINT = (
    "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}"
)


def _get_real_direct_link(sharing_link):
    pk_request = requests.get(API_ENDPOINT.format(sharing_link))

    # Returns None if the link cannot be "converted"
    return pk_request.json().get("href")


def _extract_filename(direct_link):
    for chunk in direct_link.strip().split("&"):
        if chunk.startswith("filename="):
            return chunk.split("=")[1]
    return None


def download_yadisk_link(sharing_link, filename=None):
    direct_link = _get_real_direct_link(sharing_link)
    if direct_link:
        # Try to recover the filename from the link
        filename = filename or _extract_filename(direct_link)

        download = requests.get(direct_link)
        with open(filename, "wb") as out_file:
            out_file.write(download.content)
        log('Успешно скачал "{}" в "{}"'.format(sharing_link, filename))
    else:
        log('Не удалось скачать "{}"'.format(sharing_link))
