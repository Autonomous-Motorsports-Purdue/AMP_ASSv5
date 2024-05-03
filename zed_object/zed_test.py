import pyzed.sl as sl
import cv2
import numpy as np
from matplotlib import pyplot as plt

def moving_average_wrap_around(x, w):
    # Pad the input array to ensure circular wrapping
    padded_x = np.concatenate([x[-(w//2):], x, x[:w//2]])
    
    # Compute the convolution with the padded array
    convolved = np.convolve(padded_x, np.ones(w), 'valid')
    
    # Normalize by the window size
    return convolved / w

def kde(x, sampling=np.linspace(0,255,256, dtype=int)):
    # get number of points
    n = x.shape[0]
    # determine bandwidth
    h = (4/3)**0.2 * np.std(x) * n**(-0.2)
    # generate the basis for the kde function
    bases = np.exp(-0.5 * ((sampling[:, None] - x) / h)**2) / (np.sqrt(2 * np.pi) * h)
    # then sum horizontally
    bases_summed = np.sum(bases, axis=1)
    # final normalize by the number of points in x
    kde_line = bases_summed / n
    
    return (sampling, kde_line)

def average_close_values(x, w, close):
    """
    This function takes an array x and returns an array where values in a window of size w closer than close are averaged
    """
    result = []
    for i in range(len(x)):
        window = [x[j] for j in range(max(0, i - w), min(len(x), i + w + 1))]
        close_values = [value for value in window if abs(value - x[i]) <= close]
        if close_values:
            avg = sum(close_values) // len(close_values)
            if avg not in result:  # Ensure unique values
                result.append(avg)
        else:
            result.append(x[i])
    return np.array(result, dtype=int)


def main():
    # Create a Camera object
    zed = sl.Camera()
    windowSize = 40

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.CENTIMETER
    init_params.sdk_verbose = 1 # Enable verbose mode to get more information
    init_params.depth_minimum_distance = 20

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    waitTime = 30
    while(True):
        image = sl.Mat()
        depth_for_display = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
        # A new image and depth is available if grab() returns SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve left image
            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
        
        #Display depth image using open cv
        depth_image_ocv = depth_for_display.get_data()
        small = cv2.resize(depth_image_ocv, (0,0), fx=0.25, fy=0.25)

        img = cv2.cvtColor(depth_image_ocv, cv2.COLOR_BGR2GRAY)

        small = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        #remove zero values and add window size values to the end of the array
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        #remove zero values and add window size values to the end of the array
        hist = hist[1:256] 
        num = len(hist)
        rolling2 = moving_average_wrap_around(hist[:,0], windowSize)
        print(len(rolling2))

        #Get kde of the image
        xvals, density = kde(small.flatten())
        #Plot kde so it matches histogram
        density = density*img.shape[0]*img.shape[1]

        #Add kde to rolling meanq
        combined = np.add(rolling2, density) / 2 

        grad = np.gradient(density)
        #Nomalize gradient
        grad = 20 * grad / (np.max(grad)-np.min(grad))
        #Find extrema where the gradient is around 0
        extrema = np.where(abs(grad) < 1, combined, 0)
        maximas = np.r_[True, combined[1:] > combined[:-1]] & np.r_[combined[:-1] > combined[1:], True]
        #Mark maximas on combined plot
        max_indices = average_close_values(xvals[maximas], 4, 24)
        max_indices = average_close_values(max_indices, 3, 18)
        print(max_indices)
        
        sections = [0]
        for i in range(len(max_indices)-1):
            sections.append((max_indices[i] + max_indices[i+1])//2)
        sections.append(255)

        sections = np.array(sections, dtype=int)

        print(sections)

        #Segment the image
        segments = []
        segmented = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(len(sections)-1):
            segment = np.all([img >= sections[i], img < sections[i+1]], axis=0).astype(np.uint8)*255
            segments.append(segment)
            hue = (180 * i / (len(sections) -1)) % 180  # Calculate hue value for each segment
            hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            hsv_image = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * hsv_color
            hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=segment)
            hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            segmented = cv2.add(segmented, hsv_image)


        #Combine the segments using HSV color space


        cv2.imshow('segmented', segmented)
        cv2.imshow('image', img)
        cv2.imshow('depth', depth_image_ocv)
        #Pick a random pixel in each section and store it in a list
        pixels = []
        for i in range(len(segments)):
            pixel = np.argwhere(segments[i] == 255)[0]
            pixels.append(pixel)

        
        

        
        #Sublot for histrogram stuff and for derivative stuff
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(hist)
        axs[0, 1].plot(density)
        axs[1, 0].plot(combined)
        axs[1, 0].plot(rolling2)
        
        axs[1, 0].plot(max_indices, combined[max_indices], 'rx')
        axs[1, 0].plot(sections, combined[sections], 'g.')
        #print(extrema)
        #axs[1, 0].plot(extrema)
        axs[1, 1].plot(grad)
        plt.show()
       
        k = cv2.waitKey(0) & 0xFF
        if cv2.waitKey(waitTime) & k == ord('q'):         # wait for q key to exit
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(waitTime) & k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite('zed_object\\zed_depth_standard.jpg', depth_image_ocv)
            cv2.destroyAllWindows()
    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()