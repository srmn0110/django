from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from PIL import Image #,ImageFilter
import cv2
import numpy as np
import io
from matplotlib import pyplot as plt
# import tempfile
import easyocr  # Import the easyocr library
import re
import time
from rest_framework import viewsets
from rest_framework.response import Response
from .serializers import ImageUploadSerializer


def re_search(result):
    """
    This function takes a result as input and iterates through it to identify different types of cards such as Registration Number, Aadhar Card, Pan Card, and Driving License. It returns the identified card and the rotated result.
    """
    
    card = "Not Found"
    a = 0
    pr_rotated_1 = ""
    for i in result:
        pr_rotated_1 += f" {i[0]}"
        if re.search(r'[A-Za-z]{2}[0-9TOI]{2}[A-Za-z]{2}[0-9TOI]{4}\b', i[0]):
            a = i[0][:2] + i[0][2:4].replace("T", "1").replace("O", "0").replace("I", "1")+i[0][4:6]+i[0][6:].replace("O", "0").replace("T", "1").replace("I", "1")
            print(i[0], "\t", a, "hi hello goodbye")
            card = "Registration Number"
            break
        elif re.search(r'(([0-9TOI]{4}[\sO]{1}[0-9TOI]{4}[\sO]{1}[0-9TOI]{4}))\b', i[0]):
            a = i[0].replace("T", "1").replace("O", "0").replace("I", "1")
            print(i[0], "\t", a, "hi hello goodbye")
            card = "Aadhar Card"
            break
        elif re.search(r'\b([A-Za-z]{5}[0-9TOI]{4}[A-Za-z]{1})\b', i[0]):
            a = i[0][:6] + i[0][6:10].replace("T", "1").replace("O", "0").replace("I", "1") + i[0][10:].replace("O", "0").replace("T", "1").replace("I", "1")
            print(i[0], "\t", a, "hi hello goodbye")
            card = "Pan Card"
            break
        elif re.search(r'([A-Z]{2}[0-9TOI]{14}|[A-Z]{5}[0-9TOI]{12}|[A-Z]{2}[-\s]*[0-9TOI]{12}[-\s]*[0-9TOI]{5})\b', i[0]):
            a = i[0][:2] + i[0][2:].replace("T", "1").replace("O", "0").replace("I", "1")
            print(i[0], "\t", a, "hi hello goodbye")
            card = "Driving License"
            break
        else:
            print("\t\t\t\tno\t\t",i[0])

    return a,card,pr_rotated_1

def re_search_whole(strng):
    card = "not found"
    a = 0
    i = [1,2,3]
    """
    This function performs a search on the input string to find specific patterns related to different types of cards such as Registration Number, Aadhar Card, Pan Card, and Driving License. It then processes the found patterns and returns the processed result along with the type of card found. The input parameter is the input string, and the return types are a processed string and a string indicating the type of card found.
    """
    i[0] = strng
    
    if re.search(r'[A-Za-z]{2}[0-9TOI]{2}[A-Za-z]{2}[0-9TOI]{4}\b', i[0]):
        i[0] = [re.search(r'[A-Za-z]{2}[0-9TOI]{2}[A-Za-z]{2}[0-9TOI]{4}\b', i[0]).group()][0]
        a = i[0][:2] + i[0][2:4].replace("T", "1").replace("O", "0").replace("I", "1")+i[0][4:6]+i[0][6:].replace("O", "0").replace("T", "1").replace("I", "1")
        print(i[0], "\t", a, "hi hello goodbye")
        card = "Registration Number"
        
    elif re.search(r'(([0-9TOI]{4}[\sO]{1}[0-9TOI]{4}[\sO]{1}[0-9TOI]{4}))\b', i[0][::-1]):
        aadhaar_card = [re.search(r'([0-9TOI]{4}[\sO]{1}[0-9TOI]{4}[\sO]{1}[0-9TOI]{4})\b', i[0][::-1]).group()][0]
        a = aadhaar_card.replace("T", "1").replace("O", "0").replace("I", "1")
        a = a[::-1]
        print(i[0], "\t", a, "hi hello goodbye")
        card = "Aadhar Card"
        
    elif re.search(r'\b([A-Za-z]{5}[0-9TOI]{4}[A-Za-z]{1})\b', i[0]):
        i[0] = [re.search(r'\b([A-Za-z]{5}[0-9TOI]{4}[A-Za-z]{1})\b', i[0]).group()][0]
        a = i[0][:6] + i[0][6:10].replace("T", "1").replace("O", "0").replace("I", "1") + i[0][10:].replace("O", "0").replace("T", "1").replace("I", "1")
        print(i[0], "\t", a, "hi hello goodbye")
        card = "Pan Card"
        
    elif re.search(r'([A-Z]{2}[0-9TOI]{14}|[A-Z]{5}[0-9TOI]{12}|[A-Z]{2}[-\s]*[0-9TOI]{12}[-\s]*[0-9TOI]{5})\b', i[0]):
        i[0] = [re.search(r'([A-Z]{2}[0-9TOI]{14}|[A-Z]{5}[0-9TOI]{15}|[A-Z]{2}[-\s]*[0-9TOI]{12}[-\s]*[0-9TOI]{5})\b', i[0]).group()][0]
        a = i[0][:2] + i[0][2:].replace("T", "1").replace("O", "0").replace("I", "1")
        print(i[0], "\t", a, "hi hello goodbye")
        card = "Driving License"
    return a,card


def tr(image_bytes): 
    """
    Read the image as bytes and perform text recognition on the image.

    Args:
        image_bytes: Bytes of the image to be processed.

    Returns:
        str: The recognized text from the image.

    Raises:
        None

    Examples:
        image_bytes = b'...'
        result = tr(image_bytes)
    """

        # Read the image as bytes
        # MAIN FUNCTION
    start_time = time.time()
    
    original_image = Image.open(io.BytesIO(image_bytes))

    # Convert to OpenCV format
    cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    


    # Convert the image to grayscale
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    scaled_image = cv2.resize(gray_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    threshold_image = cv2.adaptiveThreshold(
        scaled_image,
        70,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19,  # Block size
        2    # Constant subtracted from the mean
    )

    denoised_image = cv2.fastNlMeansDenoising(threshold_image, None, h=27, templateWindowSize=7, searchWindowSize=21)
    
    height, width = cv_image.shape[:2]
    # Calculate the scale factor based on the maximum allowed size
    if max(height, width) >= 2000:
        scale_factor = 0.4
    elif max(height, width) > 1500:
        scale_factor = 0.5 
    elif max(height, width) > 700:
        scale_factor = 0.78
    else:
        scale_factor = 1.0

    # # Resize the image using the calculated scale factor
    denoised_image = cv2.resize(denoised_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    
    preprocessing_complete = time.time()

    reader = easyocr.Reader(['en'])
    results = reader.readtext(denoised_image)

    reading_complete = time.time()

    average_confidence = np.mean([result[2] for result in results])
    pr_rotated_1 =""
    pr_rotated = ""
    result = [ [result[1],result[2]] for result in results]

    a=0
    a,card,pr_rotated_1 = re_search(result)
            
    unrotated_text_scan_complete = time.time()

      
    if a == 0:


        gray_image = denoised_image

        # Use Canny edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

        # Find contours in the image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the orientation angle of the text
        angle_sum = 0
        count = 0
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # You can adjust the area threshold as needed
                _, _, angle = cv2.fitEllipse(contour)
                angle_sum += angle
                count += 1

        average_angle = angle_sum / count if count > 0 else 0
        print("Average angle: {}".format(average_angle))
        
        average_angle_first = average_angle
        
        average_angle = -average_angle + 90 + 4 if average_angle > 90 else average_angle + 4
        
        
        
        
        print("converted angle: {}".format(average_angle))
        # average_angle = average_angle + 180
        # Rotate the image by the calculated angle
        rows, cols, _ = cv_image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), average_angle, 1)
        rotated_image = cv2.warpAffine(cv_image, rotation_matrix, (cols, rows))

        rows, cols, _ = cv_image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), average_angle, 1)
        rotated_image_180 = cv2.warpAffine(cv_image, rotation_matrix, (cols, rows))

        # Convert back to PIL format
        rotated_image_pil = Image.fromarray(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))

        # Use EasyOCR for text extraction with adjusted configuration on rotated image
        reader = easyocr.Reader(['en'])
        results_rotated = reader.readtext(rotated_image)
        
        
        

        
        results_rotated = [ [result[1],result[2]] for result in results_rotated]
        
        a,_,_=re_search(results_rotated)
        
        print("\n\n\nA==",a,"\n\n\n")
        
        for i in results_rotated:
            if i[0] == a:
                average_confidence = i[1]
        
        
        print([[result[0],result[1]] for result in results_rotated])

        # If average confidence is less than 0.7, rotate the image by 180 degrees and perform text extraction again
        if a==0:
            print("inverted")
            rotated_image_180 = cv2.rotate(rotated_image, cv2.ROTATE_180)
            results_rotated = reader.readtext(rotated_image_180)
            cv2.imwrite("rotated_image_180.jpg", rotated_image_180)
            cv2.imwrite("rotated_image.jpg", rotated_image)
            results_rotated = [ [result[1],result[2]] for result in results_rotated]





        # Extracted Text using EasyOCR (Rotated)
        print("\n")
        print("Extracted Text using EasyOCR (Rotated):")
        # print(results_rotated)
        print("\n")
        pr_rotated = ""
        extracted_text_easyocr_rotated = [[result[0], result[1]] for result in results_rotated]
        if a==0: #inverted changed angle
            a,card,pr_rotated = re_search(extracted_text_easyocr_rotated)

        if a == 0: #first got angle
            average_angle_first = average_angle_first - 7
            rows, cols, _ = cv_image.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), average_angle_first, 1)
            rotated_image = cv2.warpAffine(cv_image, rotation_matrix, (cols, rows))

            rows, cols, _ = cv_image.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), average_angle_first, 1)
            rotated_image_180 = cv2.warpAffine(cv_image, rotation_matrix, (cols, rows))

            # Convert back to PIL format
            rotated_image_pil = Image.fromarray(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))

            # Use EasyOCR for text extraction with adjusted configuration on rotated image
            reader = easyocr.Reader(['en'])
            results_rotated = reader.readtext(rotated_image)
            results_rotated = [ [result[1],result[2]] for result in results_rotated]
            a,card,pr_rotated = re_search(results_rotated)
            if a==0:
                print("inverted_original_angle")
                rotated_image_180 = cv2.rotate(rotated_image, cv2.ROTATE_180)
                results_rotated = reader.readtext(rotated_image_180)
                cv2.imwrite("rotated_image_180.jpg", rotated_image_180)
                cv2.imwrite("rotated_image.jpg", rotated_image)
                results_rotated = [ [result[1],result[2]] for result in results_rotated]
                a,card,pr_rotated = re_search(results_rotated)
    rotation_complete = time.time()










    if a == 0 :
        print("\n\n\n",pr_rotated_1,"\n\n\n")
        a,card = re_search_whole(pr_rotated_1)
            

    if a==0:
        print(pr_rotated)
        a,card = re_search_whole(pr_rotated)
    last = time.time()

    print("\n")
    print("\n")
    print(a,"==a" )
    print("\n")
    print("\n")    
    print(preprocessing_complete - start_time,"preprocessing_complete - start_time")
    print(reading_complete - preprocessing_complete,"reading_complete - preprocessing_complete")
    print(unrotated_text_scan_complete - reading_complete,"unrotated_text_scan_complete - reading_complete")
    print(rotation_complete - unrotated_text_scan_complete,"rotation_complete - unrotated_text_scan_complete")
    print(last - rotation_complete,"last - rotation_complete")


    end_time = time.time()
    print("Time taken: {} seconds".format(end_time - start_time))
    if a!=0:
        a = a.upper()
    else:
        a = "not found"
    return a,card



@csrf_exempt
def index(request):
    if request.method == 'POST':
        # Get the input image from the request
        input_image = request.FILES.get('image')

        # Read the image as bytes
        image_bytes = input_image.read()
        a="not found"
        a = tr(image_bytes)



        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        # Pass the base64-encoded image and ID number to the template
        context = {'image': f"data:image/{input_image.content_type};base64,{encoded_image}",
                   'id_number': a,}

        # Pass the extracted_text variable to the index.html template
        return render(request, 'index.html', context)

    # If it's not a POST request, render the index.html template
    return render(request, 'index.html')



class ImageViewSet(viewsets.ViewSet):
    def create(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        input_image = serializer.validated_data['image']
        image_bytes = input_image.read()
        result,card = tr(image_bytes)

        return Response({'id_number': result,"card":card})
