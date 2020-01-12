import os
import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(FILE_PATH, "../..")
DATA_PATH = os.path.join(ROOT_PATH, "data")
CONF_PATH = os.path.join(ROOT_PATH, "conf")
RES_PATH = os.path.join(ROOT_PATH, "res")
SRC_PATH = os.path.join(ROOT_PATH, "src")
LOG_PATH = os.path.join(ROOT_PATH, "log")
PIC_PATH = os.path.join(ROOT_PATH, "pic")
sys.path.insert(0, SRC_PATH)
if os.path.exists(RES_PATH) == False:
    os.mkdir(RES_PATH)
if os.path.exists(LOG_PATH) == False:
    os.mkdir(LOG_PATH)
