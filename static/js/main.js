// main.js

document.addEventListener('DOMContentLoaded', function () {
    // Obtener los elementos de los cards
    const trainingCard = document.getElementById('trainingCard');
    const uploadCard = document.getElementById('uploadCard');
    const realtimeCard = document.getElementById('realtimeCard');
    const home = document.getElementById('home');

    // Función para redirigir a la página especificada
    function redirectToPage(page) {
        window.location.href = `/${page}`;
    }

    // Agregar manejadores de eventos de clic a los cards
    home.addEventListener('click', function () {
        redirectToPage('');
    });

    trainingCard.addEventListener('click', function () {
        redirectToPage('training');
    });

    uploadCard.addEventListener('click', function () {
        redirectToPage('upload');
    });

    realtimeCard.addEventListener('click', function () {
        redirectToPage('realtime');
    });
});
