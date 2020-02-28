# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import json
import requests
from termcolor import colored


def query(text, url):
    print(colored("↑↑↑↑ ", "red") + colored(text, "cyan"))
    print(colored("↓↓↓↓ ", "green") + colored(
        json.loads(
            requests.post(url, json={"text": text}).text
        ).get("tags"), "cyan"))


if __name__ == "__main__":
    query(sys.argv[1], "http://192.168.110.10:2020/decode")
