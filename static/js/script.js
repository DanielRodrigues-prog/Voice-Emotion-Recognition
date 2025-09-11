// Aguarda o carregamento completo do HTML
document.addEventListener('DOMContentLoaded', () => {
    // Seleciona os elementos da página
    const audioFileInput = document.getElementById('audioFileInput');
    const predictButton = document.getElementById('predictButton');
    const resultDiv = document.getElementById('result');

    // Adiciona um "ouvinte" para o clique no botão
    predictButton.addEventListener('click', () => {
        const file = audioFileInput.files[0];

        // Verifica se um arquivo foi selecionado
        if (!file) {
            resultDiv.textContent = 'Por favor, selecione um arquivo de áudio.';
            return;
        }

        // Cria um objeto FormData para enviar o arquivo
        const formData = new FormData();
        formData.append('file', file);

        // Mostra uma mensagem de "carregando"
        resultDiv.textContent = 'Analisando... 🧠';

        // Faz a requisição para a nossa API Flask usando fetch
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // Se a resposta não for OK, lança um erro
            if (!response.ok) {
                throw new Error(`Erro do servidor: ${response.status}`);
            }
            return response.json(); // Converte a resposta para JSON
        })
        .then(data => {
            // Exibe o resultado na página
            resultDiv.textContent = `Emoção Detectada: ${data.emotion}`;
        })
        .catch(error => {
            // Exibe qualquer erro que tenha ocorrido
            console.error('Erro:', error);
            resultDiv.textContent = `Erro ao analisar o áudio. Tente novamente. (${error.message})`;
        });
    });
});