document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('originalImage').src = e.target.result;

            document.getElementById('result').style.display = 'block';
            document.getElementById('outputImage').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/run', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    document.getElementById('license').textContent = `License: ${result.label}`;
    document.getElementById('score').textContent = `Score: ${result.score}`;
    document.getElementById('outputImage').src = `data:image/jpeg;base64,${result.image}`;

    document.getElementById('result').style.display = 'block';
    document.getElementById('outputImage').style.display = 'block';
});