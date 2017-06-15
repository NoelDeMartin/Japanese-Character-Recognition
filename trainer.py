# coding=utf-8

import struct
import os
from PIL import Image, ImageEnhance

datasets = [
	('07', 11288),
	('08', 11288),
	('09', 11287), # TODO ナ(NA) on Sheet 2672 is missing
	('10', 11288),
	('11', 11288),
	('12', 11287), # TODO リ(RI) on Sheet 2708 is missing
	('13', 4233),
];

for dataset in datasets:
	filename = 'data/raw/ETL1/ETL1C_' + dataset[0]

	japanese_katakana = []
	for vowel in 'AIUEO':
		japanese_katakana.append(vowel)
		for consonant in 'KSTNHMYRWN':
			japanese_katakana.append(consonant + vowel)

	file = open(filename, 'r')

	for i in range(dataset[1]):
		file.seek(i * 2052)
		data = struct.unpack('>H2sH6BI4H4B4x2016s4x', file.read(2052))
		character = data[1].strip()
		raw_image = Image.frombytes('F', (64, 63), data[18], 'bit', 4)
		raw_image = raw_image.convert('P')
		enhancer = ImageEnhance.Brightness(raw_image)

		path = 'data/katakana/{}'.format(character)
		if not os.path.exists(path):
			os.makedirs(path)

		enhancer.enhance(16).save('{}/{}.png'.format(path, i), 'PNG')

	file.close()