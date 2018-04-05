# coding=utf-8

import zipfile
import struct
import random
import os

import numpy as np

from PIL import Image, ImageEnhance

# If this is set to True, only アイウエオカキク characters will be extracted,
# which make for a faster training time with better accuracy at the cost
# of learning less characters.
REDUCED_TRAINING_SET = True

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

		if REDUCED_TRAINING_SET:
			datasets = [('07', 11288)]
		else:
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

			with open(relative_path('katakana/data'), 'w') as data_file:

				categories_file.write('category,katakana_character')
				classification = []
				categories = []
				count = 0

				datasets_count = len(datasets)
				for dataset in range(datasets_count):
					dataset_suffix, dataset_size = datasets[dataset]

					with open(relative_path('raw/ETL1/ETL1C_' + dataset_suffix), 'r') as file:

						for i in range(dataset_size):
							file.seek(i * 2052 + 2)
							character = file.read(2).strip()

							if character not in categories:
								categories.append(character)
								categories_file.write('\n{},{}'.format(categories.index(character), character))

							file.seek(i * 2052 + 33)
							prepare_image(data_file, file.read(2016))

							classification.append((count, categories.index(character)))
							count = count + 1

							if i % 1000 == 0:
								print 'Unpacking dataset {}/{} - {}% ...'.format(
											dataset + 1, datasets_count, int((float(i) / dataset_size) * 100))

			with open(relative_path('katakana/classification.csv'), 'w') as classification_file:

				classification_file.write('position,category')
				random.shuffle(classification)
				for position, category in classification:
					classification_file.write('\n{},{}'.format(position, category))

	print 'Katakana unpacked!'

def prepare_image(data_file, image_data):

	image = Image.frombytes('F', (64, 63), image_data, 'bit', 4)
	image = image.convert('P')
	image = ImageEnhance.Brightness(image).enhance(40)
	image = image.resize((76, 76))
	image = image.crop((6, 6, 70, 70))

	new_img = Image.new('1', (64, 64))
	new_img.paste(image, (0, 0))
	new_img = Image.eval(new_img, lambda x: not x)

	data_file.write(np.packbits(np.array(new_img.getdata())))

# Runtime

if __name__ == '__main__':
	main()
