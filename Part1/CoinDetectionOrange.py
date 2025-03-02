import numpy as np
import cv2
import os

def load_and_preprocess(imagePath):
    originalImage = cv2.imread(imagePath)
    if originalImage is None:
        print(f"Error loading image: {imagePath}")
        return None, None, None
    
    imgHeight, imgWidth = originalImage.shape[:2]
    maxDimension = max(imgHeight, imgWidth)
    scaleFactor = 700 / maxDimension if maxDimension > 700 else 1
    resizedImage = cv2.resize(originalImage, (0, 0), fx=scaleFactor, fy=scaleFactor)
    
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    thresholdImage = cv2.adaptiveThreshold(
        blurredImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return resizedImage, thresholdImage, scaleFactor

def find_coin_contours(binaryImage, scaleFactor):
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coinCandidates = []
    
    for contour in contours:
        contourPerimeter = cv2.arcLength(contour, True)
        contourArea = cv2.contourArea(contour)
        
        if contourPerimeter == 0:
            continue
            
        circularity = 4 * np.pi * (contourArea / (contourPerimeter ** 2))
        minCoinArea = 500 * (scaleFactor ** 2)
        
        if 0.7 < circularity < 1.2 and contourArea > minCoinArea:
            coinCandidates.append(contour)
    
    return coinCandidates

def save_processed_image(baseImage, contours, outputPath, mode="outline"):
    resultImage = baseImage.copy()
    
    if mode == "outline":
        cv2.drawContours(resultImage, contours, -1, (0, 0, 255), 2)
    elif mode == "mask":
        mask = np.zeros(baseImage.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        resultImage = np.zeros_like(baseImage)
        resultImage[np.where(mask == 255)] = (0, 140, 255)  # Orange color for objects
    
    cv2.imwrite(outputPath, resultImage)

def extract_individual_coins(baseImage, contours, outputDir):
    extractedCoins = []
    
    for idx, contour in enumerate(contours):
        # Get bounding circle coordinates
        (centerX, centerY), radius = cv2.minEnclosingCircle(contour)
        center = (int(centerX), int(centerY))
        radius = int(radius)
        
        # Create circular mask
        mask = np.zeros_like(baseImage, dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        maskedCoin = cv2.bitwise_and(baseImage, mask)
        
        # Calculate crop coordinates
        startX = center[0] - radius
        startY = center[1] - radius
        endX = center[0] + radius
        endY = center[1] + radius
        
        croppedCoin = maskedCoin[startY:endY, startX:endX]
        extractedCoins.append(croppedCoin)
        
        # Save individual coin
        coinPath = os.path.join(outputDir, f"coin_{idx+1}.jpg")
        cv2.imwrite(coinPath, croppedCoin)
    
    return extractedCoins

def process_image_batch(inputDir, outputDir):
    os.makedirs(outputDir, exist_ok=True)
    
    # Create organized output directories
    individualDir = os.path.join(outputDir, "ImageIndividual")
    segmentedDir = os.path.join(outputDir, "ImageSegmented")
    outlineDir = os.path.join(outputDir, "ImageOutline")
    
    for dirPath in [individualDir, segmentedDir, outlineDir]:
        os.makedirs(dirPath, exist_ok=True)
    
    for fileName in os.listdir(inputDir):
        if not fileName.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
            
        inputPath = os.path.join(inputDir, fileName)
        baseName = os.path.splitext(fileName)[0]
        
        # Create output paths
        outlinePath = os.path.join(outlineDir, f"{baseName}_outline.jpg")
        segmentedPath = os.path.join(segmentedDir, f"{baseName}_mask.jpg")
        coinsFolder = os.path.join(individualDir, f"{baseName}_coins")
        
        os.makedirs(coinsFolder, exist_ok=True)
        
        # Process image
        resizedImage, binaryImage, scale = load_and_preprocess(inputPath)
        if resizedImage is None:
            continue
        
        detectedCoins = find_coin_contours(binaryImage, scale)
        
        # Save outputs to respective directories
        save_processed_image(resizedImage, detectedCoins, outlinePath, "outline")
        save_processed_image(resizedImage, detectedCoins, segmentedPath, "mask")
        extract_individual_coins(resizedImage, detectedCoins, coinsFolder)
        
        print(f"Processed {fileName}: Found {len(detectedCoins)} coins")

# Run the processing
process_image_batch("Input", "Output")