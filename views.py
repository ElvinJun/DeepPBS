from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from django.http import HttpResponse
from django.conf import settings
from django.core.files import File
import logging
import subprocess
import random
import os
import time
logger = logging.getLogger('django')


def save_dir():
    LOCAL_TIME = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    FILES_DIR = os.path.join(r'files', LOCAL_TIME)
    if not os.path.exists(FILES_DIR):
        os.makedirs(FILES_DIR)

    else:
        FILES_DIR = FILES_DIR + '-' + str(random.randint(1, 1000))
        os.makedirs(FILES_DIR)
    SAVED_FILES_DIR = os.path.join(FILES_DIR, 'CA_info')
    os.makedirs(SAVED_FILES_DIR)
    return SAVED_FILES_DIR

# SAVED_FILES_DIR = save_dir()
# files = os.listdir(SAVED_FILES_DIR)
# for file in files:
#     file_pathname = os.path.join(SAVED_FILES_DIR, file)
#     os.unlink(file_pathname)

# Create your views here.
def render_home_template(request):

    return render(request, 'home.html')

def render_home_template1(request):
    files = os.listdir(SAVED_FILES_DIR)
    return render(request, 'download.html', {'files': files})

def home(request):

    return render(request, 'home.html')


def download(request, filename):
    file_pathname = os.path.join(SAVED_FILES_DIR.replace('CA_info', 'backbone'), filename)

    with open(file_pathname, 'rb') as f:
        file = File(f)

        response = HttpResponse(file.chunks(),
                                content_type='APPLICATION/OCTET-STREAM')
        response['Content-Disposition'] = 'attachment; filename=' + filename
        response['Content-Length'] = os.path.getsize(file_pathname)
    # os.unlink(file_pathname)
    return response


def upload(request):
    global SAVED_FILES_DIR
    SAVED_FILES_DIR = save_dir()
    files = request.FILES.getlist('filename')
    if not files:
        return render_home_template(request)


    for file in files:
        destination = open(SAVED_FILES_DIR + '/' + file.name, 'wb+')
        for chunk in file.chunks():
            destination.write(chunk)

        destination.close()
    shell = 'python D:/python/webserver/fileoperation/model.py ' + SAVED_FILES_DIR
    child = subprocess.Popen(shell,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = child.communicate()
    result = str(stderr, encoding='utf-8')  # return result
    logger.info(result)
    return render_home_template1(request)


def index(request):
    return render(request, 'index.html')
