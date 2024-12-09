// JS para interactividad simple, como un cambio en la visibilidad de las secciones.
document.addEventListener('DOMContentLoaded', () => {
    const paragraphs = document.querySelectorAll('.parrafo');

    paragraphs.forEach(paragraph => {
        paragraph.addEventListener('click', () => {
            paragraph.classList.toggle('expand');
        });
    });
});

// Para expandir o contraer el texto de los párrafos al hacer clic
const style = document.createElement('style');
style.textContent = `
    .parrafo.expand {
        white-space: normal;
        overflow: visible;
    }
    .parrafo {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
`;
document.head.append(style);
