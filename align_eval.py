import cv2
import time
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import os
import glob
import shutil
import pandas as pd
import pytesseract
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import math

# Toggle debugging
DEBUG = True

def debug_print(message):
    if DEBUG:
        print(message)

# Paths for input/output
pdf_path = 'pdfInputs'  # Adjust the path to your PDFs
png_output_path = 'outputImages'  # Where the PNG files will be saved
reslts_folder_path = 'results'

# Convert PDFs to PNGs
pdf_files = glob.glob(os.path.join(pdf_path, '*.pdf'))

# Create the output_pngs folder if it doesn't exist, or clear it if it does
if not os.path.exists(png_output_path):
    os.makedirs(png_output_path)
else:
    shutil.rmtree(png_output_path)
    os.makedirs(png_output_path)

# Create the results folder if it doesn't exist, or clear it if it does
if not os.path.exists(reslts_folder_path):
    os.makedirs(reslts_folder_path)

debug_print(f"Processing PDF files in {pdf_path}")

for pdf_file in pdf_files:
    debug_print(f"Converting {pdf_file} to PNGs")
    images = convert_from_path(pdf_file)
    for i, image in enumerate(images):
        image_path = os.path.join(png_output_path, f"{os.path.splitext(os.path.basename(pdf_file))[0]}_page_{i+1}.png")
        image.save(image_path, 'PNG')

# Load converted PNG files for processing
form_images = glob.glob(os.path.join(png_output_path, '*.png'))

# Features list for comparison
features_list = [512, 1024, 1589, 2048, 4096, 5192, 10245, 15689, 25849, 35469]

# Results storage for each method
results_orb = {'processing_times': [], 'cumulative_times': [], 'matches': [], 'success': 0, 'features': [], 'aligned_images': []}
results_sift = {'processing_times': [], 'cumulative_times': [], 'matches': [], 'success': 0, 'features': [], 'aligned_images': []}
results_freak = {'processing_times': [], 'cumulative_times': [], 'matches': [], 'success': 0, 'features': [], 'aligned_images': []}
results_freak_sift = {'processing_times': [], 'cumulative_times': [], 'matches': [], 'success': 0, 'features': [], 'aligned_images': []}

# Load the CSV file with the keypoint coordinates and check phrases
csv_file_path = 'keypointsLocations.csv'  # Adjust the path to your CSV file
keypoints_df = pd.read_csv(csv_file_path)

# Convert the DataFrame into a dictionary for easier access
keypoints_dict = {
    'x1': keypoints_df['x1'].tolist(),
    'y1': keypoints_df['y1'].tolist(),
    'x2': keypoints_df['x2'].tolist(),
    'y2': keypoints_df['y2'].tolist(),
    'checkPhraseList': keypoints_df['checkPhraseList'].tolist()
}

# Function to align images using ORB
def align_images_orb(template_img, scanned_img, max_features):
    orb = cv2.ORB_create(max_features)

    # Find the keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(template_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(scanned_img, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches
    img_matches = cv2.drawMatches(template_img, keypoints1, scanned_img, keypoints2, matches[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography to warp image
    height, width = template_img.shape
    aligned_img = cv2.warpPerspective(scanned_img, h, (width, height))

    return img_matches, aligned_img, len(matches)

# Function to align images using ORB for keypoints and FREAK for descriptors
def align_images_freak(template_img, scanned_img, max_features):
    # Initialize ORB detector
    orb = cv2.ORB_create(max_features)

    # Find the keypoints with ORB
    keypoints1 = orb.detect(template_img, None)
    keypoints2 = orb.detect(scanned_img, None)

    # Initialize FREAK descriptor
    freak = cv2.xfeatures2d.FREAK_create()

    # Compute the descriptors with FREAK
    keypoints1, descriptors1 = freak.compute(template_img, keypoints1)
    keypoints2, descriptors2 = freak.compute(scanned_img, keypoints2)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches
    img_matches = cv2.drawMatches(template_img, keypoints1, scanned_img, keypoints2, matches[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography to warp image
    height, width = template_img.shape
    aligned_img = cv2.warpPerspective(scanned_img, h, (width, height))

    return img_matches, aligned_img, len(matches)

# Function to align images using SIFT
def align_images_sift(template_img, scanned_img, max_features):
    sift = cv2.SIFT_create(max_features)

    # Find the keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(template_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(scanned_img, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches
    img_matches = cv2.drawMatches(template_img, keypoints1, scanned_img, keypoints2, matches[:500], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography to warp image
    height, width = template_img.shape
    aligned_img = cv2.warpPerspective(scanned_img, h, (width, height))

    return img_matches, aligned_img, len(matches)

# New function to align images using FREAK until 15000 features and switch to SIFT if FREAK fails
def align_images_freak_sift(template_img, scanned_img, max_features):
    if max_features <= 15000:
        # Use FREAK for features <= 15000
        return align_images_freak(template_img, scanned_img, max_features)
    else:
        # Try FREAK first
        img_matches, aligned_img, matches = align_images_freak(template_img, scanned_img, max_features)
        
        # Check if the alignment was successful
        if aligned_img is not None:
            aligned = check_alignment(aligned_img, keypoints_dict)
            if aligned:
                return img_matches, aligned_img, matches  # Return FREAK results if successful
        
        # If FREAK fails or alignment is unsuccessful, switch to SIFT
        return align_images_sift(template_img, scanned_img, max_features)

def check_alignment(aligned_image, keypoints_dict):
    num_regions_to_check = len(keypoints_dict['x1'])
    
    # Sequentially check each keypoint
    for i in range(num_regions_to_check):
        # Extract the region of interest based on the coordinates
        roi = aligned_image[keypoints_dict['y1'][i]:keypoints_dict['y2'][i],
                            keypoints_dict['x1'][i]:keypoints_dict['x2'][i]]
        
        # Use Tesseract to extract text from the region
        extracted_text = pytesseract.image_to_string(roi, config='--psm 6')
        extracted_text = extracted_text.lower().strip()
        debug_print(f"Extracted Text{i}:{extracted_text}")

        # Get the list of possible correct phrases
        check_phrases = keypoints_dict['checkPhraseList'][i].split(', ')
        check_phrases = [phrase.lower().strip() for phrase in check_phrases]

        # Check if the extracted text matches any of the check phrases for this keypoint
        if not any(phrase in extracted_text for phrase in check_phrases):
            return False  # If one fails, return False immediately

    return True  # All keypoints matched

# Main loop to generate results and comparison PDF
pdf_filename = f"results\\Alignment_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
with PdfPages(pdf_filename) as pdf:
    template_img = cv2.imread('template.png', 0)
    
    for form_image_path in form_images:
        debug_print(f"Processing form image: {form_image_path}")
        scanned_img = cv2.imread(form_image_path, 0)

        form_results = {}  # Clear form_results for each form

        for algo_name, align_func, results in zip(
            ['ORB', 'SIFT', 'FREAK', 'FREAK+SIFT'], 
            [align_images_orb, align_images_sift, align_images_freak, align_images_freak_sift], 
            [results_orb, results_sift, results_freak, results_freak_sift]
        ):
            first_success = None
            cumulative_time = 0

            for features in features_list:
                if first_success is None:
                    debug_print(f"Aligning using {algo_name} with {features} features")
                    start_time = time.time()
                    img_matches, aligned_img, matches = align_func(template_img, scanned_img, features)
                    end_time = time.time()
                    processing_time = end_time - start_time
                    cumulative_time += processing_time

                    if aligned_img is not None:
                        aligned = check_alignment(aligned_img, keypoints_dict)
                        debug_print(f"{algo_name} - Features: {features}, Matches: {matches}, Aligned: {aligned}")

                        if aligned:
                            first_success = features
                            results['processing_times'].append(processing_time)
                            results['cumulative_times'].append(cumulative_time)
                            results['matches'].append(matches)
                            results['success'] += 1
                            results['features'].append(features)
                            results['aligned_images'].append(aligned_img)

                            # Store the result in form_results for later use
                            form_results[algo_name] = (features, matches, aligned_img, cumulative_time)
                            break
                    else:
                        debug_print(f"Not enough matches for {algo_name} with {features} features.")

            if first_success is None:
                # Store failure in form_results if alignment was unsuccessful
                form_results[algo_name] = (None, None, None, cumulative_time)

        # Determine the number of algorithms
        num_algorithms = len(form_results)

        # Calculate the number of rows and columns needed
        num_cols = 2  # Set the number of columns
        num_rows = math.ceil(num_algorithms / num_cols)  # Calculate the number of rows

        # Generate PDF page with results for each algorithm
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
        fig.suptitle(f"Alignment Results for {os.path.basename(form_image_path)}")

        # Flatten axs array if it's multidimensional (necessary for easy iteration)
        axs = axs.flatten() if num_rows * num_cols > 1 else [axs]

        # Display the aligned images or failure messages for each algorithm
        for i, (algo_name, result) in enumerate(form_results.items()):
            features, matches, aligned_img, _ = result

            if aligned_img is not None:
                axs[i].imshow(aligned_img, cmap='gray')
                axs[i].set_title(f"{algo_name} Aligned (Features: {features})")
            else:
                axs[i].text(0.5, 0.5, f"{algo_name} Alignment Failed", horizontalalignment='center', verticalalignment='center')
            axs[i].axis('off')

        # Hide any remaining empty subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        # Save the page to the PDF
        pdf.savefig(fig)
        plt.close(fig)

    # Generate comparison graphs for metrics
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    # Hide the last subplot (bottom-right)
    fig.delaxes(axs[2, 1])  # This removes the sixth plot

    # Average Processing Time
    axs[0, 0].bar(['ORB', 'SIFT', 'FREAK', 'FREAK+SIFT'], 
                  [np.mean(results_orb['processing_times']), np.mean(results_sift['processing_times']), np.mean(results_freak['processing_times']), np.mean(results_freak_sift['processing_times'])], 
                  color=['red', 'blue', 'green', 'purple'])
    axs[0, 0].set_title('Average Processing Time per Attempt')
    axs[0, 0].set_ylabel('Time (seconds)')

    # Average Cumulative Time
    axs[0, 1].bar(['ORB', 'SIFT', 'FREAK', 'FREAK+SIFT'], 
                  [np.mean(results_orb['cumulative_times']), np.mean(results_sift['cumulative_times']), np.mean(results_freak['cumulative_times']), np.mean(results_freak_sift['cumulative_times'])], 
                  color=['red', 'blue', 'green', 'purple'])
    axs[0, 1].set_title('Average Total Processing Time per Form')
    axs[0, 1].set_ylabel('Time (seconds)')

    # Success Rate
    axs[1, 0].bar(['ORB', 'SIFT', 'FREAK', 'FREAK+SIFT'], 
                  [results_orb['success'] / len(form_images) * 100, 
                   results_sift['success'] / len(form_images) * 100, 
                   results_freak['success'] / len(form_images) * 100,
                   results_freak_sift['success'] / len(form_images) * 100], 
                  color=['red', 'blue', 'green', 'purple'])
    axs[1, 0].set_title('Alignment Success Rate')
    axs[1, 0].set_ylabel('Success Rate (%)')

    # Average Matches
    axs[1, 1].bar(['ORB', 'SIFT', 'FREAK', 'FREAK+SIFT'], 
                  [np.mean(results_orb['matches']), np.mean(results_sift['matches']), np.mean(results_freak['matches']), np.mean(results_freak_sift['matches'])], 
                  color=['red', 'blue', 'green', 'purple'])
    axs[1, 1].set_title('Average Matches per Attempt')
    axs[1, 1].set_ylabel('Number of Matches')

    # Average Features
    axs[2, 0].bar(['ORB', 'SIFT', 'FREAK', 'FREAK+SIFT'], 
                  [np.mean(results_orb['features']), np.mean(results_sift['features']), np.mean(results_freak['features']), np.mean(results_freak_sift['features'])], 
                  color=['red', 'blue', 'green', 'purple'])
    axs[2, 0].set_title('Average Features per Attempt')
    axs[2, 0].set_ylabel('Number of Features')

    pdf.savefig(fig)
    plt.close(fig)

print(f"Results saved in {pdf_filename}")
