/*
 * This JavaScript doesn't do anything. The file exists just to demonstrate
 * including static assets from the HTML in themes.
 */

// função para colocar a legenda sobre as imagens
window.onload = function() {
  var imagens = document.getElementsByTagName( "img" );
  for (var figura in imagens) {
    if (imagens[figura].title){
      conteudo = document.createElement( 'p' );
      conteudo.innerHTML = imagens[figura].title;
      // conteudo.style = "color:gray; text-align: center; font-size: 10pt;";
      conteudo.classList.add("legendas");
      imagens[figura].parentNode.insertBefore(conteudo, imagens[figura].nextSibling);
    }
  }
}

// função para escoder e ver o menu novamente, usando o botão na barra do topo
function verEsconderMenu() {
  var omenu = document.getElementById("navegacao");
  var oconteudo = document.getElementById("conteudo");
  omenu.style.display = omenu.style.display === 'none' ? '' : 'none';
  if (omenu.style.display === 'none') {
    document.getElementById("conteudo").style.left = '0px'
  } else {
    document.getElementById("conteudo").style.left = '230px'
  }
}
