import cv2
import numpy as np
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = "./output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_and_match_features(image1, image2):
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    # Use Brute-Force Matcher with L2 norm 
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Match descriptors
    matches = matcher.match(descriptors1, descriptors2)
    
    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return keypoints1, keypoints2, matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):

    # Draw matches
    matched_image = cv2.drawMatches(
        image1, keypoints1, 
        image2, keypoints2, 
        matches[:50],  # Draw only the top 50 matches for clarity
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return matched_image

def create_panorama(image1, image2):

    keypoints1, keypoints2, matches = detect_and_match_features(image1, image2)
    
    # Extract location of good matches
    source_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Compute homography
    homography_matrix, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
    
    # Warp image
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    
    # Create a panorama canvas
    panorama_width = width1 + width2
    panorama_height = max(height1, height2)
    
    panorama = cv2.warpPerspective(image1, homography_matrix, (panorama_width, panorama_height))
    panorama[0:height2, 0:width2] = image2
    
    return panorama

def main():
    # Load images
    image_left = cv2.imread("./input/temp_2.jpeg")
    image_right = cv2.imread("./input/temp_1.jpeg")

    # Resizing for consistency
    image_left = cv2.resize(image_left, (800, 600))
    image_right = cv2.resize(image_right, (800, 600))

    # Detect keypoints and matches
    keypoints1, keypoints2, matches = detect_and_match_features(image_left, image_right)

    # Draw keypoints on images
    image_left_keypoints = cv2.drawKeypoints(image_left, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_right_keypoints = cv2.drawKeypoints(image_right, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints Image 1", image_left_keypoints)
    cv2.imshow("Keypoints Image 2", image_right_keypoints)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "keypoints_1.jpg"), image_left_keypoints)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "keypoints_2.jpg"), image_right_keypoints)

    # Draw matches between the two images
    matched_image = draw_matches(image_left, keypoints1, image_right, keypoints2, matches)
    cv2.imshow("Matches Between Images", matched_image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "matches.jpg"), matched_image)

    # Stitch images
    panorama = create_panorama(image_left, image_right)

    # Show and save output
    cv2.imshow("Panorama", panorama)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "panorama_output.jpg"), panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()