import random
from random import randint
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from raindrop.raindrop import raindrop
"""
This script generate the Drop on the images
Author: Chia-Tse, Chang

"""


def CheckCollision(DropList):
	"""
	This function handle the collision of the drops
	"""
	
	listFinalDrops = []
	Checked_list = []
	list_len = len(DropList)
	# because latter raindrops in raindrop list should has more colision information
	# so reverse list	
	DropList.reverse()
	drop_key = 1
	for drop in DropList:
		# if the drop has not been handle	
		if drop.getKey() not in Checked_list:			
			# if drop has collision with other drops
			if drop.getIfColli():
				# get collision list
				collision_list = drop.getCollisionList()
				# first get radius and center to decide how  will the collision do
				final_x = drop.getCenters()[0] * drop.getRadius()
				final_y = drop.getCenters()[1]  * drop.getRadius()
				tmp_devide = drop.getRadius()
				final_R = drop.getRadius()  * drop.getRadius()
				for col_id in collision_list:				
					Checked_list.append(col_id)
					# list start from 0				
					final_x += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[0]
					final_y += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[1]
					tmp_devide += DropList[list_len - col_id].getRadius()
					final_R += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getRadius() 
				final_x = int(round(final_x/tmp_devide))
				final_y = int(round(final_y/tmp_devide))
				final_R = int(round(math.sqrt(final_R)))
				# rebuild drop after handled the collisions
				newDrop = raindrop(drop_key, (final_x, final_y), final_R)
				drop_key = drop_key+1
				listFinalDrops.append(newDrop)
			# no collision
			else:
				drop.setKey(drop_key)
				drop_key = drop_key+1
				listFinalDrops.append(drop)			
	

	return listFinalDrops


def np_label(binary_array, connectivity=2):
	"""
    Simple connected component labeling using BFS for binary 2D array.
    Returns labeled array and number of labels.
    """
	from collections import deque

	labeled = np.zeros_like(binary_array, dtype=int)
	label_num = 0
	h, w = binary_array.shape

	if connectivity == 4:
		neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
	else:  # connectivity == 8
		neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

	for i in range(h):
		for j in range(w):
			if binary_array[i, j] and labeled[i, j] == 0:
				label_num += 1
				q = deque()
				q.append((i, j))
				labeled[i, j] = label_num
				while q:
					x, y = q.popleft()
					for dx, dy in neighbors:
						nx, ny = x + dx, y + dy
						if 0 <= nx < h and 0 <= ny < w and binary_array[nx, ny] and labeled[nx, ny] == 0:
							labeled[nx, ny] = label_num
							q.append((nx, ny))
	return labeled, label_num

def generateDrops(imagePath, cfg, inputLabel=None):
    """
    This function generates raindrops on an image with optional label input.
    """
    maxDrop = cfg["maxDrops"]
    minDrop = cfg["minDrops"]
    drop_num = randint(minDrop, maxDrop)
    maxR = cfg["maxR"]
    minR = cfg["minR"]
    ifReturnLabel = cfg["return_label"]
    edge_ratio = cfg["edge_darkratio"]

    # Always work in RGB to avoid 2D arrays and channel mismatches
    PIL_bg_img = Image.open(imagePath).convert('RGB')
    bg_img = np.asarray(PIL_bg_img)
    # 2D label map regardless of channels
    label_map = np.zeros(bg_img.shape[:2], dtype=np.uint8)
    imgh, imgw = bg_img.shape[:2]

    # random drops position
    ran_pos = [(int(random.random() * imgw), int(random.random() * imgh)) for _ in range(drop_num)]
    listRainDrops = []

    # Create raindrops
    if inputLabel is None:
        for key, pos in enumerate(ran_pos):
            key += 1
            radius = random.randint(minR, maxR)
            drop = raindrop(key, pos, radius)
            listRainDrops.append(drop)
    else:
        arrayLabel = np.asarray(inputLabel)
        condition = (arrayLabel[:, :, 0] > cfg["label_thres"])
        label = np.where(condition, 1, 0)
        label_part, label_nums = np_label(label, connectivity=2)
        for idx in range(label_nums):
            i = idx + 1
            label_index = np.argwhere(label_part == i)
            U, D = np.min(label_index[:, 0]), np.max(label_index[:, 0]) + 1
            L, R = np.min(label_index[:, 1]), np.max(label_index[:, 1]) + 1
            cur_alpha = arrayLabel[U:D, L:R, 0].copy()
            cur_label = (cur_alpha > cfg["label_thres"]) * 1
            centerxy = (L, U)
            drop = raindrop(idx, centerxy=centerxy, input_alpha=cur_alpha, input_label=cur_label)
            listRainDrops.append(drop)

    # Handle collision
    collisionNum = len(listRainDrops)
    listFinalDrops = list(listRainDrops)
    loop = 0
    if inputLabel is None:
        while collisionNum > 0:
            loop += 1
            listFinalDrops = list(listFinalDrops)
            collisionNum = len(listFinalDrops)
            label_map = np.zeros_like(label_map)
            for drop in listFinalDrops:
                (ix, iy) = drop.getCenters()
                radius = drop.getRadius()
                ROI_WL, ROI_WR = 2 * radius, 2 * radius
                ROI_HU, ROI_HD = 3 * radius, 2 * radius
                if (iy - 3 * radius) < 0:
                    ROI_HU = iy
                if (iy + 2 * radius) > imgh:
                    ROI_HD = imgh - iy
                if (ix - 2 * radius) < 0:
                    ROI_WL = ix
                if (ix + 2 * radius) > imgw:
                    ROI_WR = imgw - ix

                drop_label = drop.getLabelMap()
                if (label_map[iy, ix] > 0):
                    col_ids = np.unique(label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL:ix + ROI_WR])
                    col_ids = col_ids[col_ids != 0]
                    drop.setCollision(True, col_ids)
                    label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL:ix + ROI_WR] = drop_label[
                        3 * radius - ROI_HU:3 * radius + ROI_HD,
                        2 * radius - ROI_WL:2 * radius + ROI_WR] * drop.getKey()
                else:
                    label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL:ix + ROI_WR] = drop_label[
                        3 * radius - ROI_HU:3 * radius + ROI_HD,
                        2 * radius - ROI_WL:2 * radius + ROI_WR] * drop.getKey()
                    collisionNum -= 1

            if collisionNum > 0:
                listFinalDrops = CheckCollision(listFinalDrops)

    # Add alpha for the edge of drops
    alpha_map = np.zeros_like(label_map).astype(np.float64)
    if inputLabel is None:
        for drop in listFinalDrops:
            (ix, iy) = drop.getCenters()
            radius = drop.getRadius()
            ROI_WL, ROI_WR = 2 * radius, 2 * radius
            ROI_HU, ROI_HD = 3 * radius, 2 * radius
            if (iy - 3 * radius) < 0:
                ROI_HU = iy
            if (iy + 2 * radius) > imgh:
                ROI_HD = imgh - iy
            if (ix - 2 * radius) < 0:
                ROI_WL = ix
            if (ix + 2 * radius) > imgw:
                ROI_WR = imgw - ix

            drop_alpha = drop.getAlphaMap()
            alpha_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL:ix + ROI_WR] += drop_alpha[
                3 * radius - ROI_HU:3 * radius + ROI_HD,
                2 * radius - ROI_WL:2 * radius + ROI_WR]
    else:
        for drop in listFinalDrops:
            (ix, iy) = drop.getCenters()
            drop_alpha = drop.getAlphaMap()
            h, w = drop_alpha.shape
            alpha_map[iy:iy + h, ix:ix + w] += drop_alpha[:h, :w]

    # Normalize and apply Gaussian blur using Pillow
    if np.max(alpha_map) > 0:
        alpha_map = alpha_map / np.max(alpha_map) * 255.0
    alpha_map_img = Image.fromarray(np.uint8(alpha_map))
    alpha_map_img = alpha_map_img.filter(ImageFilter.GaussianBlur(radius=10))
    alpha_map = np.array(alpha_map_img).astype(np.float64)
    if np.max(alpha_map) > 0:
        alpha_map = alpha_map / np.max(alpha_map) * 255.0

    # Apply drops to image
    PIL_bg_img = PIL_bg_img.copy()
    for drop in listFinalDrops:
        (ix, iy) = drop.getCenters()
        if inputLabel is None:
            radius = drop.getRadius()
            ROIU, ROID = max(iy - 3 * radius, 0), min(iy + 2 * radius, imgh)
            ROIL, ROIR = max(ix - 2 * radius, 0), min(ix + 2 * radius, imgw)
        else:
            h, w = drop.getLabelMap().shape
            ROIU, ROID, ROIL, ROIR = iy, iy + h, ix, ix + w

        tmp_bg = bg_img[ROIU:ROID, ROIL:ROIR, :]
        drop.updateTexture(tmp_bg)
        tmp_alpha_map = alpha_map[ROIU:ROID, ROIL:ROIR]
        tmp_output = np.asarray(drop.getTexture()).astype(np.float64)[:, :, -1]
        tmp_alpha_map = tmp_alpha_map * (tmp_output / 255)
        tmp_alpha_map = Image.fromarray(tmp_alpha_map.astype('uint8'))

        edge = ImageEnhance.Brightness(drop.getTexture()).enhance(edge_ratio)
        if inputLabel is None:
            PIL_bg_img.paste(edge, (ix - 2 * radius, iy - 3 * radius), tmp_alpha_map)
            PIL_bg_img.paste(drop.getTexture(), (ix - 2 * radius, iy - 3 * radius), drop.getTexture())
        else:
            PIL_bg_img.paste(edge, (ix, iy), tmp_alpha_map)
            PIL_bg_img.paste(drop.getTexture(), (ix, iy), drop.getTexture())

    if ifReturnLabel:
        output_label = np.array(alpha_map)
        output_label[output_label > 0] = 1
        output_label = Image.fromarray(output_label.astype('uint8'))
        return PIL_bg_img, output_label

    return PIL_bg_img
