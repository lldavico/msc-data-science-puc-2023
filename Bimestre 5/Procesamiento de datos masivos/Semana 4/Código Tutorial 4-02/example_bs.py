from bs4 import BeautifulSoup

# abrir el archivo HTML local
with open("web_simple.html") as fp:
    # crear un objeto BeautifulSoup con el contenido del archivo
    soup = BeautifulSoup(fp, "html.parser")

# ahora puedes manipular el objeto "soup"
# por ejemplo, puedes imprimir el contenido del archivo HTML
# print(soup.prettify())

# encontrar el primer tag <h1> en el archivo HTML e imprimir su texto
# h1_tag = soup.find("h1")
# print(h1_tag.text)

# encontrar todos los tags <li> en el archivo HTML
# li_tags = soup.find_all("li")
# for li_tag in li_tags:
#     print(li_tag.text)

# encontrar el primer tag <a> en el archivo HTML
# a_tag = soup.find("a")
# print(a_tag["href"])

# encontrar el tag con id "fecha" en el archivo HTML
# fecha_tag = soup.find(id="fecha")
# print(fecha_tag.text)

# encontrar el tag con clase "precipitaciones" en el archivo HTML
# precipitaciones_tag = soup.find(class_="precipitaciones")
# print(precipitaciones_tag.text)

# encontrar el tag <li> con clase "precipitaciones" en el archivo HTML
li_tag = soup.find("li", class_="precipitaciones")
print(li_tag.text)
