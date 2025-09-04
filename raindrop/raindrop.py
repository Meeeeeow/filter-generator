import math
import numpy as np
from PIL import Image, ImageFilter

class raindrop():
    def __init__(self, key, centerxy=None, radius=None, input_alpha=None, input_label=None):
        if input_label is None:
            self.key = key
            self.ifcol = False
            self.col_with = []
            self.center = centerxy
            self.radius = radius
            self.type = "default"
            self.labelmap = np.zeros((self.radius * 5, self.radius * 4))
            self.alphamap = np.zeros((self.radius * 5, self.radius * 4))
            self.background = None
            self.texture = None
            self._create_label()
            self.use_label = False
        else:
            self.key = key
            assert input_alpha is not None, "Please also input the alpha map"
            self.alphamap = input_alpha
            self.labelmap = input_label
            self.ifcol = False
            self.col_with = []
            h, w = self.labelmap.shape
            self.center = centerxy
            self.radius = min(w // 4, h // 4)
            self.background = None
            self.texture = None
            self.use_label = True

    def setCollision(self, col, col_with):
        self.ifcol = col
        self.col_with = col_with

    def updateTexture(self, bg):
        # bg: numpy array (H, W, 3)
        fg = Image.fromarray(np.uint8(bg))
        fg = fg.filter(ImageFilter.GaussianBlur(radius=5))
        fg = np.array(fg)

        # Resize alpha map to match fg size
        alpha_img = Image.fromarray(np.uint8(self.alphamap))
        alpha_img = alpha_img.resize((fg.shape[1], fg.shape[0]))  # (width, height)
        alpha_array = np.array(alpha_img)

        # Combine bg and alpha
        tmp = np.expand_dims(alpha_array, axis=-1)
        tmp = np.concatenate((fg, tmp), axis=2)

        self.texture = Image.fromarray(tmp.astype('uint8'), 'RGBA')
        self.texture = self.texture.transpose(Image.FLIP_TOP_BOTTOM)

    def _create_label(self):
        if self.type == "default":
            self._createDefaultDrop()
        elif self.type == "splash":
            self._createSplashDrop()

    def _createDefaultDrop(self):
        # Draw simple circle and ellipse on labelmap using NumPy
        h, w = self.labelmap.shape
        yy, xx = np.ogrid[:h, :w]
        cx, cy = self.radius * 2, self.radius * 3

        # Circle
        circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= self.radius ** 2
        self.labelmap[circle_mask] = 128

        # Ellipse
        ellipse_mask = ((xx - cx) ** 2) / (self.radius ** 2) + ((yy - cy) ** 2) / ((1.3 * math.sqrt(3) * self.radius) ** 2) <= 1
        self.labelmap[ellipse_mask] = 128

        # Alpha map using Pillow Gaussian blur
        alpha_img = Image.fromarray(np.uint8(self.labelmap))
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=10))
        self.alphamap = np.array(alpha_img).astype(np.float64)
        self.alphamap = self.alphamap / np.max(self.alphamap) * 255.0

        # Set label map binary
        self.labelmap[self.labelmap > 0] = 1

    def _createSplashDrop(self):
        pass

    def setKey(self, key):
        self.key = key

    def getLabelMap(self):
        return self.labelmap

    def getAlphaMap(self):
        return self.alphamap

    def getTexture(self):
        return self.texture

    def getCenters(self):
        return self.center

    def getRadius(self):
        return self.radius

    def getKey(self):
        return self.key

    def getIfColli(self):
        return self.ifcol

    def getCollisionList(self):
        return self.col_with

    def getUseLabel(self):
        return self.use_label
