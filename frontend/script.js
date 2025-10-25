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
    toggle.textContent = "Show file upload instructions ▼";
} else {
    info.style.display = "block";
    toggle.textContent = "Hide file upload instructions ▲";
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

// Collect voice descriptions
const voiceDescriptions = [];
for (let i = 1; i <= numActors; i++) {
    const desc = form[`actor${i}`].value || "";
    voiceDescriptions.push(desc);
}
formData.append("voiceDescriptions", JSON.stringify(voiceDescriptions));

// Example backend endpoint
const response = await fetch("/api/generate-audio", {
    method: "POST",
    body: formData,
});

if (!response.ok) {
    alert("Error generating audio.");
    return;
}

const blob = await response.blob();
const audioUrl = URL.createObjectURL(blob);

document.getElementById("output").style.display = "block";
document.getElementById("audioPlayer").src = audioUrl;
});

// Load sample script for demo page
document.addEventListener("DOMContentLoaded", () => {
    const scriptDisplay = document.getElementById("scriptDisplay");

    fetch("TestingMultitalk/tomorrow.txt")
        .then(response => {
            if (!response.ok) throw new Error("Failed to load script file.");
            return response.text();
        })
        .then(text => {
            scriptDisplay.textContent = text;
        })
        .catch(error => {
            scriptDisplay.textContent = "Error loading demo script: " + error.message;
        });
});