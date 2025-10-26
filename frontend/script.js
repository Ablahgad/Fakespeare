const form = document.getElementById("audioForm");
const numActorsInput = document.getElementById("numActors");
const voiceInputsContainer = document.getElementById("voiceInputs");

// Generate actor description boxes dynamically
function updateVoiceInputs() {
const numActors = parseInt(numActorsInput.value) || 1;
voiceInputsContainer.innerHTML = "";

for (let i = 1; i <= numActors; i++) {
    const label = document.createElement("label");
    label.textContent = `Actor ${i} Description:`;

    const input = document.createElement("input");
    input.type = "text";
    input.name = `actor${i}`;
    input.placeholder = `e.g. Actor ${i}: calm female voice`;

    const wrapper = document.createElement("div");
    wrapper.classList.add("actor-description");
    wrapper.appendChild(label);
    wrapper.appendChild(input);

    voiceInputsContainer.appendChild(wrapper);
}
}

// Initialize default
updateVoiceInputs();
numActorsInput.addEventListener("change", updateVoiceInputs);

// Toggle info section
const toggle = document.getElementById("toggleInfo");
const info = document.getElementById("fileInfo");

toggle.addEventListener("click", () => {
if (info.style.display === "block") {
    info.style.display = "none";
    toggle.textContent = "Hark! What file type can I upload? ▼";
} else {
    info.style.display = "block";
    toggle.textContent = "Away, away, ye file type instructions! ▲";
}
});

// Handle form submission
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("textFile");
    if (!fileInput.files.length) {
        alert("Please upload a text file.");
        return;
    }

    const numActors = parseInt(numActorsInput.value);
    const formData = new FormData();
    formData.append("textFile", fileInput.files[0]);
    formData.append("numActors", numActors);
});

// Collect voice descriptions
const voiceDescriptions = [];
for (let i = 1; i <= numActors; i++) {
    const desc = form[`actor${i}`].value || "";
    voiceDescriptions.push(desc);
}
// formData.append("voiceDescriptions", JSON.stringify(voiceDescriptions));


// Ensure this script is included with <script src="script.js"></script> in your HTML
document.getElementById("submit-button").addEventListener("click", async () => {
  const fileInput = document.getElementById("textFile");
  const file = fileInput.files[0];

  const button = document.getElementById("submit-button");
  button.innerText = "Loading...";

  
  if (!file) {
    alert("Please select a file first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    // Send the file to the Flask backend
    const response = await fetch("http://127.0.0.1:5000/generate_audio", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Request failed");

    // Get the audio as a blob
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);

    const audioSource = document.getElementById("audio-output");
    audioSource.src = audioUrl;

    const audioPlayer = document.getElementById("audioPlayer");
    audioPlayer.load();
    audioPlayer.play();

    const output = document.getElementById("output");
    output.style.display = "block";
    button.innerText = "Regenerate Audio";


  } catch (err) {
    console.error("Error:", err);
    alert("There was an error generating the audio. See console for details.");
  }
});

// // Load sample script for demo page
// document.addEventListener("DOMContentLoaded", () => {
//     const scriptDisplay = document.getElementById("scriptDisplay");

//     fetch("../TestingMultitalk/tomorrow.txt")
//         .then(response => {
//             if (!response.ok) throw new Error("Failed to load script file.");
//             return response.text();
//         })
//         .then(text => {
//             scriptDisplay.textContent = text;
//         })
//         .catch(error => {
//             scriptDisplay.textContent = "Error loading demo script: " + error.message;
//         });
// });