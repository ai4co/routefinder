document.addEventListener('DOMContentLoaded', function() {
    const codeContainers = document.querySelectorAll('.code-container');
    
    codeContainers.forEach(container => {
      const copyButton = container.querySelector('.copy-button');
      const codeContent = container.querySelector('.code-content');
      const copyTooltip = container.querySelector('.copy-tooltip');
  
      copyButton.addEventListener('click', function() {
        const textArea = document.createElement('textarea');
        textArea.value = codeContent.textContent.trim();
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
  
        copyButton.classList.add('copied');
        copyTooltip.textContent = 'Copied!';
        
        setTimeout(() => {
          copyButton.classList.remove('copied');
          copyTooltip.textContent = 'Copy to clipboard';
        }, 2000);
      });
    });
  });