function openCamera() {
    // Access the device camera for video stream
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            videoStream = stream;
            const video = document.createElement('video');
            video.srcObject = stream;
            video.autoplay = true;
            video.style.maxWidth = '100%'; // Set maximum width to maintain aspect ratio

            // Create a container for the video element
            const videoContainer = document.createElement('div');
            videoContainer.classList.add('video-container');
            videoContainer.style.width = '80%'; // Set width of video container
            videoContainer.style.margin = 'auto'; // Center the container horizontally
            videoContainer.appendChild(video);
            
            // Set width of video container
            videoContainer.appendChild(video);

            // Append the video container to the pestDetection card
            const pestDetectionCard = document.querySelector('.pestDetection');
            pestDetectionCard.innerHTML = ''; // Clear existing content
            pestDetectionCard.appendChild(videoContainer);

            // Create capture button
            const captureButton = document.createElement('button');
            captureButton.textContent = 'Capture Image';
            captureButton.classList.add('capture-button');
            captureButton.addEventListener('click', captureImage);
            pestDetectionCard.appendChild(captureButton);
        })
        .catch(err => {
            console.error('Error accessing the camera: ', err);
        });
}


function captureImage() {
    if (videoStream) {
        const videoTrack = videoStream.getVideoTracks()[0];
        const imageCapture = new ImageCapture(videoTrack);

        imageCapture.takePhoto()
            .then(blob => {
                // Create a new window to display the captured image
                const newWindow = window.open('', '_blank');
                const image = document.createElement('img');
                image.src = URL.createObjectURL(blob);
                image.style.width = '100%'; // Adjust size if necessary

                // Append the image to the new window
                newWindow.document.body.appendChild(image);

                // Stop the video stream
                videoTrack.stop();
            })
            .catch(err => {
                console.error('Error capturing image: ', err);
            });
    }
}



// Add click event listener to the camera icon
document.getElementById('cameraIcon').addEventListener('click', openCamera);
    
    