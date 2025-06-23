#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uvicorn
from fastapi import FastAPI
import numpy as np
import cv2
import json

from template_extractor import TemplateExtractor
from classifier import Classifier

extractor = TemplateExtractor()
classifier_obj = Classifier()

app = FastAPI()

@app.get("/resilience/extract_template")
async def template_extraction(image_path: str = ""):
    """ Extract Biometric Template Endpoint. """
    template_data = extractor.extract(image_path=image_path)
    return {'data': template_data.tolist()}

@app.get("/resilience/compare_templates")
async def compare_two_templates_endpoint(template_1: str = "", template_2: str = ""):
    """ Compare 2 feature templates """
    try:
        bin_data_1 = json.loads(template_1)
        bin_data_2 = json.loads(template_2)

        if len(bin_data_1) != len(bin_data_2):
            return {'similarity': "ERROR", 'comment': "Templates size mismatch"}

        bin_array_1 = np.array(bin_data_1, dtype=float)
        bin_array_2 = np.array(bin_data_2, dtype=float)

        similarity = extractor.compare(bin_array_1, bin_array_2)

        return {'similarity': float(similarity), 'comment': 'OK'}

    except Exception as e:
        return {'similarity': "ERROR", 'comment': str(e)}

@app.get("/resilience/compare_images")
async def compare_two_images_endpoint(image_path_1: str = "", image_path_2: str = ""):
    """ Compare 2 Images by their filenames."""
    try:
        template_1 = extractor.extract(image_path=image_path_1)
        template_2 = extractor.extract(image_path=image_path_2)

        similarity = extractor.compare(template_1, template_2)

        return {
            'image_path_1': image_path_1,
            'image_path_2': image_path_2,
            'similarity': float(similarity),
            'comment': 'OK'
        }

    except Exception as e:
        return {
            'image_path_1': image_path_1,
            'image_path_2': image_path_2,
            'similarity': "ERROR",
            'comment': str(e)
        }

@app.get("/detection/detect")
async def classify_image(image_path: str = ""):
    """ Extract Adversarial Decision Endpoint. """
    try:
        score, decision = classifier_obj.extract(image_path=image_path)
        return {
            'image_path': image_path,
            'score': float(score),
            'decision': bool(decision),
            'comment': 'OK'
        }

    except Exception as e:
        return {
            'image_path': image_path,
            'score': "ERROR",
            'decision': "ERROR",
            'comment': str(e)
        }

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=7007)
