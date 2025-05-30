import os
import shutil
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ['KAGGLE_USERNAME'] = 'humbertogiuri'
os.environ['KAGGLE_KEY'] = 'c43f989f9a9a2fe5da091df56739cec1'

# Inicializa a API do Kaggle
api = KaggleApi()
api.authenticate()

# Diretórios
dataset_slug = 'kmader/skin-cancer-mnist-ham10000'
download_dir = 'kaggle_ham10000'
extracted_dir = os.path.join(download_dir, 'extracted')
output_dir = 'HAM10000_by_class'

# Baixa o dataset
print("🔽 Baixando dataset HAM10000...")
api.dataset_download_files(dataset_slug, path=download_dir, unzip=True)
print("✅ Download e extração concluídos.")

# Caminhos dos arquivos extraídos
metadata_path = os.path.join(download_dir, 'HAM10000_metadata.csv')
images_dir_1 = os.path.join(download_dir, 'ham10000_images_part_1')
images_dir_2 = os.path.join(download_dir, 'ham10000_images_part_2')

# Lê o CSV de metadados
df = pd.read_csv(metadata_path)

# Cria o diretório de saída
os.makedirs(output_dir, exist_ok=True)

# Contador de imagens por classe
class_counts = {}

# Organiza as imagens por classe
for idx, row in df.iterrows():
    image_id = row['image_id']
    lesion_type = row['dx']

    # Caminhos possíveis (part 1 ou part 2)
    src1 = os.path.join(images_dir_1, image_id + '.jpg')
    src2 = os.path.join(images_dir_2, image_id + '.jpg')

    src_image_path = src1 if os.path.exists(src1) else src2
    if not os.path.exists(src_image_path):
        print(f"[⚠️] Imagem não encontrada: {image_id}.jpg")
        continue

    # Cria a pasta da classe
    class_folder = os.path.join('data', output_dir, lesion_type)
    os.makedirs(class_folder, exist_ok=True)

    # Copia a imagem
    dst_image_path = os.path.join(class_folder, image_id + '.jpg')
    shutil.copy2(src_image_path, dst_image_path)

    class_counts[lesion_type] = class_counts.get(lesion_type, 0) + 1

# Resumo final
print("\n📁 Organização por classe concluída. Total por classe:")
for c, count in class_counts.items():
    print(f"  - {c}: {count} imagens")
