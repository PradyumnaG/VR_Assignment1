# import numpy as np
# import cv2
# import os

# def load_and_preprocess(imagePath):
#     originalImage = cv2.imread(imagePath)
#     if originalImage is None:
#         print(f"Error loading image: {imagePath}")
#         return None, None, None
    
#     imgHeight, imgWidth = originalImage.shape[:2]
#     maxDimension = max(imgHeight, imgWidth)
#     scaleFactor = 700 / maxDimension if maxDimension > 700 else 1
#     resizedImage = cv2.resize(originalImage, (0, 0), fx=scaleFactor, fy=scaleFactor)
    
#     grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
#     blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
#     thresholdImage = cv2.adaptiveThreshold(
#         blurredImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 11, 2
#     )
    
#     return resizedImage, thresholdImage, scaleFactor

# def find_coin_contours(binaryImage, scaleFactor):
#     contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     coinCandidates = []
    
#     for contour in contours:
#         contourPerimeter = cv2.arcLength(contour, True)
#         contourArea = cv2.contourArea(contour)
        
#         if contourPerimeter == 0:
#             continue
            
#         circularity = 4 * np.pi * (contourArea / (contourPerimeter ** 2))
#         minCoinArea = 500 * (scaleFactor ** 2)
        
#         if 0.7 < circularity < 1.2 and contourArea > minCoinArea:
#             coinCandidates.append(contour)
    
#     return coinCandidates

# def save_processed_image(baseImage, contours, outputPath, mode="outline"):
#     resultImage = baseImage.copy()
    
#     if mode == "outline":
#         cv2.drawContours(resultImage, contours, -1, (0, 0, 255), 2)
#     elif mode == "mask":
#         mask = np.zeros(baseImage.shape[:2], dtype=np.uint8)
#         cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
#         resultImage = np.zeros_like(baseImage)
#         resultImage[np.where(mask == 255)] = (0, 140, 255)  # Orange color for objects
    
#     cv2.imwrite(outputPath, resultImage)

# def extract_individual_coins(baseImage, contours, outputDir):
#     extractedCoins = []
    
#     for idx, contour in enumerate(contours):
#         (centerX, centerY), radius = cv2.minEnclosingCircle(contour)
#         center = (int(centerX), int(centerY))
#         radius = int(radius)
        
#         mask = np.zeros_like(baseImage, dtype=np.uint8)
#         cv2.circle(mask, center, radius, (255, 255, 255), -1)
#         maskedCoin = cv2.bitwise_and(baseImage, mask)
        
#         startX = center[0] - radius
#         startY = center[1] - radius
#         endX = center[0] + radius
#         endY = center[1] + radius
        
#         croppedCoin = maskedCoin[startY:endY, startX:endX]
#         extractedCoins.append(croppedCoin)
        
#         coinPath = os.path.join(outputDir, f"coin_{idx+1}.jpg")
#         cv2.imwrite(coinPath, croppedCoin)
    
#     return extractedCoins

# def process_image_batch(inputDir, outputDir):
#     os.makedirs(outputDir, exist_ok=True)
    
#     individualDir = os.path.join(outputDir, "ImageIndividual")
#     segmentedDir = os.path.join(outputDir, "ImageSegmented")
#     outlineDir = os.path.join(outputDir, "ImageOutline")
    
#     for dirPath in [individualDir, segmentedDir, outlineDir]:
#         os.makedirs(dirPath, exist_ok=True)
    
#     for fileName in os.listdir(inputDir):
#         if not fileName.lower().endswith((".png", ".jpg", ".jpeg")):
#             continue
            
#         inputPath = os.path.join(inputDir, fileName)
#         baseName = os.path.splitext(fileName)[0]
        
#         outlinePath = os.path.join(outlineDir, f"{baseName}_outline.jpg")
#         segmentedPath = os.path.join(segmentedDir, f"{baseName}_mask.jpg")
#         coinsFolder = os.path.join(individualDir, f"{baseName}_coins")
        
#         os.makedirs(coinsFolder, exist_ok=True)
        
#         resizedImage, binaryImage, scale = load_and_preprocess(inputPath)
#         if resizedImage is None:
#             continue
        
#         detectedCoins = find_coin_contours(binaryImage, scale)
        
#         save_processed_image(resizedImage, detectedCoins, outlinePath, "outline")
#         save_processed_image(resizedImage, detectedCoins, segmentedPath, "mask")
#         extract_individual_coins(resizedImage, detectedCoins, coinsFolder)
        
#         print(f"Processed {fileName}: Found {len(detectedCoins)} coins")

# process_image_batch("Input", "Output")


import numpy as np
import cv2
import os

def load_and_preprocess(imagePath):

    # Load the image
    originalImage = cv2.imread(imagePath)
    if originalImage is None:
        print(f"Error loading image: {imagePath}")
        return None, None, None
    
    # Get image dimensions and calculate scaling factor
    imgHeight, imgWidth = originalImage.shape[:2]
    maxDimension = max(imgHeight, imgWidth)
    scaleFactor = 700 / maxDimension if maxDimension > 700 else 1
    
    # Resize the image
    resizedImage = cv2.resize(originalImage, (0, 0), fx=scaleFactor, fy=scaleFactor)
    
    # Convert the image to grayscale
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    
    # Apply adaptive thresholding to create a binary image
    thresholdImage = cv2.adaptiveThreshold(
        blurredImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return resizedImage, thresholdImage, scaleFactor

def find_coin_contours(binaryImage, scaleFactor):
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coinCandidates = []
    
    for contour in contours:
        # Calculate perimeter and area of the contour
        contourPerimeter = cv2.arcLength(contour, True)
        contourArea = cv2.contourArea(contour)
        
        # Skip if the perimeter is zero
        if contourPerimeter == 0:
            continue
            
        # Calculate circularity of the contour
        circularity = 4 * np.pi * (contourArea / (contourPerimeter ** 2))
        
        # Define minimum area for a coin based on the scaling factor
        minCoinArea = 500 * (scaleFactor ** 2)
        
        # Filter contours based on circularity and area
        if 0.7 < circularity < 1.2 and contourArea > minCoinArea:
            coinCandidates.append(contour)
    
    return coinCandidates

def save_processed_image(baseImage, contours, outputPath, mode="outline"):

    resultImage = baseImage.copy()
    
    if mode == "outline":
        # Draw contours on the image in outline mode
        cv2.drawContours(resultImage, contours, -1, (0, 0, 255), 2)
    elif mode == "mask":
        # Create a binary mask and fill the detected coins with orange color
        mask = np.zeros(baseImage.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        resultImage = np.zeros_like(baseImage)
        resultImage[np.where(mask == 255)] = (0, 140, 255)  # Orange color for objects
    
    # Save the processed image
    cv2.imwrite(outputPath, resultImage)

def extract_individual_coins(baseImage, contours, outputDir):
    
    extractedCoins = []
    
    for idx, contour in enumerate(contours):
        # Get the bounding circle for the contour
        (centerX, centerY), radius = cv2.minEnclosingCircle(contour)
        center = (int(centerX), int(centerY))
        radius = int(radius)
        
        # Create a circular mask for the coin
        mask = np.zeros_like(baseImage, dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        
        # Apply the mask to isolate the coin
        maskedCoin = cv2.bitwise_and(baseImage, mask)
        
        # Calculate the bounding box for the coin
        startX = center[0] - radius
        startY = center[1] - radius
        endX = center[0] + radius
        endY = center[1] + radius
        
        # Crop the coin from the image
        croppedCoin = maskedCoin[startY:endY, startX:endX]
        extractedCoins.append(croppedCoin)
        
        # Save the cropped coin as a separate file
        coinPath = os.path.join(outputDir, f"coin_{idx+1}.jpg")
        cv2.imwrite(coinPath, croppedCoin)
    
    return extractedCoins

def process_image_batch(inputDir, outputDir):
    
    # Create the output directory if it doesn't exist
    os.makedirs(outputDir, exist_ok=True)
    
    # Create subdirectories for individual coins, segmented images, and outlined images
    individualDir = os.path.join(outputDir, "ImageIndividual")
    segmentedDir = os.path.join(outputDir, "ImageSegmented")
    outlineDir = os.path.join(outputDir, "ImageOutline")
    
    for dirPath in [individualDir, segmentedDir, outlineDir]:
        os.makedirs(dirPath, exist_ok=True)
    
    # Process each image in the input directory
    for fileName in os.listdir(inputDir):
        # Skip non-image files
        if not fileName.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
            
        # Construct the full input path
        inputPath = os.path.join(inputDir, fileName)
        baseName = os.path.splitext(fileName)[0]
        
        # Define output paths for outlined images, segmented images, and individual coins
        outlinePath = os.path.join(outlineDir, f"{baseName}_outline.jpg")
        segmentedPath = os.path.join(segmentedDir, f"{baseName}_mask.jpg")
        coinsFolder = os.path.join(individualDir, f"{baseName}_coins")
        
        # Create the directory for individual coins
        os.makedirs(coinsFolder, exist_ok=True)
        
        # Load and preprocess the image
        resizedImage, binaryImage, scale = load_and_preprocess(inputPath)
        if resizedImage is None:
            continue
        
        # Find coin contours in the binary image
        detectedCoins = find_coin_contours(binaryImage, scale)
        
        # Save the outlined and segmented images
        save_processed_image(resizedImage, detectedCoins, outlinePath, "outline")
        save_processed_image(resizedImage, detectedCoins, segmentedPath, "mask")
        
        # Extract and save individual coins
        extract_individual_coins(resizedImage, detectedCoins, coinsFolder)
        
        # Print the number of coins detected in the image
        print(f"Processed {fileName}: Found {len(detectedCoins)} coins")

# Process all images in the "Input" directory and save results in the "Output" directory
process_image_batch("Input", "Output")