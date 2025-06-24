import os
import shutil
import zipfile
import requests

ARCHIVE_URL = "https://github.com/nerel-ds/NEREL/archive/refs/heads/master.zip"
ARCHIVE_NAME = "NEREL-master.zip"
EXTRACTED_FOLDER = "NEREL-master/NEREL-v1.1"
TARGET_DIR = os.path.join("nerel_dataset", "data", "NEREL-v1.1")


def download_file(url, filename):
    print(f"Скачиваю архив {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Архив сохранён как {filename}")


def extract_nerel_v1_1(zip_path, target_dir):
    print(f"Распаковываю только папку NEREL-v1.1 в {target_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = [m for m in zip_ref.namelist() if m.startswith(EXTRACTED_FOLDER + '/')]
        if os.path.exists(target_dir):
            print(f"Удаляю существующую папку {target_dir} ...")
            shutil.rmtree(target_dir)
        for member in members:
            filename = member[len(EXTRACTED_FOLDER)+1:]
            if not filename:
                continue  # Пропускаем саму папку
            dest = os.path.join(target_dir, filename)
            if member.endswith('/'):
                # Это директория, создаём её
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zip_ref.open(member) as source, open(dest, "wb") as target:
                shutil.copyfileobj(source, target)
    print("Распаковка завершена.")


def main():
    download_file(ARCHIVE_URL, ARCHIVE_NAME)
    extract_nerel_v1_1(ARCHIVE_NAME, TARGET_DIR)
    os.remove(ARCHIVE_NAME)
    print("Готово! NEREL-v1.1 скачан и распакован.")


if __name__ == "__main__":
    main() 
