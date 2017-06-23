# coding=utf-8

import zipfile
import struct
import random
import os

import numpy as np

from PIL import Image, ImageEnhance

def main():
	extract_zip()
	unpack_katakana()

# Method definitions

def relative_path(path):
	return os.path.dirname(os.path.realpath(__file__)) + '/' + path

def extract_zip():
	output_dir = relative_path('raw/ETL1')
	if not os.path.exists(output_dir):
		print 'Extracting raw/ETL1.zip...'

		with zipfile.ZipFile(relative_path('raw/ETL1.zip'), 'r') as file:
			file.extractall(relative_path('raw'))

	print 'raw/ETL1.zip extracted!'

def unpack_katakana():
	output_dir = relative_path('katakana')
	if not os.path.exists(output_dir):
		print 'Unpacking katakana...'

		os.makedirs(output_dir)

		datasets = [
			('07', 11288),
			('08', 11288),
			('09', 11287), # TODO ナ(NA) on Sheet 2672 is missing
			('10', 11288),
			('11', 11288),
			('12', 11287), # TODO リ(RI) on Sheet 2708 is missing
			('13', 4233),
		]

		with open(relative_path('katakana/categories.csv'), 'w') as categories_file:

			categories_file.write('category,katakana_character')
			classification = []
			categories = []

			datasets_count = len(datasets)
			for dataset in range(datasets_count):
				dataset_suffix, dataset_size = datasets[dataset]

				with open(relative_path('raw/ETL1/ETL1C_' + dataset_suffix), 'r') as file:

					for i in range(dataset_size):
						file.seek(i * 2052)
						data = struct.unpack('>H2sH6BI4H4B4x2016s4x', file.read(2052))
						character = data[1].strip()
						image = prepare_image(Image.frombytes('F', (64, 63), data[18], 'bit', 4))

						output_dir = relative_path('katakana/images/{}'.format(character))
						if not os.path.exists(output_dir):
							os.makedirs(output_dir)

						image.save('{}/{}.png'.format(output_dir, i), 'PNG')

						if character not in categories:
							categories.append(character)
							categories_file.write('\n{},{}'.format(categories.index(character), character))

						classification.append(('images/{}/{}.png'.format(character, i), categories.index(character)))

						if i % 1000 == 0:
							print 'Unpacking dataset {}/{} - {}% ...'.format(
										dataset + 1, datasets_count, int((float(i) / dataset_size) * 100))

			with open(relative_path('katakana/classification.csv'), 'w') as classification_file:

				classification_file.write('relative_path,category')
				random.shuffle(classification)
				for path, category in classification:
					classification_file.write('\n{},{}'.format(path, category))

	print 'Katakana unpacked!'

def prepare_image(raw_image):
	image = raw_image.resize((64, 64), Image.LANCZOS)
	image = image.convert('RGB')
	image = ImageEnhance.Brightness(image).enhance(16)

	data = np.array(image)
	data = 255 - data

	return Image.fromarray(data, mode='RGB')

# Runtime

if __name__ == '__main__':
	main()