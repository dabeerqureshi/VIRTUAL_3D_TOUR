document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("tour-form");
    const placeNameInput = document.getElementById("place_name");
    const placeImageInput = document.getElementById("place_image");
    const placeDescInput = document.getElementById("place_desc");
    const errorMessage = document.getElementById("error_message");
    const loadingMessage = document.getElementById("loading");
    const resultVideo = document.getElementById("output");
    const videoSource = document.getElementById("video_source");

    if (!form || !errorMessage || !loadingMessage || !resultVideo || !videoSource) {
        console.error("❌ ERROR: One or more required elements are missing from the DOM!");
        return;
    }

    form.addEventListener("submit", async function (event) {
        event.preventDefault();
        console.log("✅ Submit button clicked!"); // Debugging log

        // Reset previous states
        errorMessage.style.display = "none";
        resultVideo.style.display = "none";

        const placeName = placeNameInput.value.trim();
        const placeImage = placeImageInput.files[0];
        const placeDesc = placeDescInput.value.trim();

        let formData = new FormData();
        let inputCount = 0;

        if (placeName) {
            formData.append("place_name", placeName);
            inputCount++;
        }
        if (placeImage) {
            formData.append("place_image", placeImage);
            inputCount++;
        }
        if (placeDesc) {
            formData.append("place_desc", placeDesc);
            inputCount++;
        }

        // Ensure exactly **one** input is provided at a time
        if (inputCount !== 1) {
            errorMessage.innerText = "⚠️ Please provide only one input at a time.";
            errorMessage.style.display = "block";
            return;
        }

        loadingMessage.style.display = "block"; // Show loading message

        try {
            const response = await fetch("/get_3d_visual", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            console.log("✅ Server Response:", result); // Debugging log

            loadingMessage.style.display = "none"; // Hide loading message

            if (response.ok && result["3D_video"]) { 
                let fileUrl = result["3D_video"];
                console.log("🎥 3D Video URL received:", fileUrl);

                // Update the video player
                videoSource.src = fileUrl;
                resultVideo.style.display = "block";
                resultVideo.load();
                resultVideo.play();
            } else {
                errorMessage.innerText = result["error"] || "❌ Invalid response from server.";
                errorMessage.style.display = "block";
            }
        } catch (error) {
            console.error("🚨 Error:", error);
            errorMessage.innerText = "❌ An error occurred! Please check your network connection and try again.";
            errorMessage.style.display = "block";
        }
    });
});
