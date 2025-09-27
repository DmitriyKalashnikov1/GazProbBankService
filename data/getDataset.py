from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import re
from time import sleep
import random
import os
import csv


linkP1 = "https://www.banki.ru/services/responses/bank/gazprombank/?page="
linkP2 = "&is_countable=on"

baseLink = 'https://www.banki.ru'

if os.path.exists("links.txt"):
    with open("links.txt") as file:
        cartLinks = file.readlines()
else:
    cartLinks = []

# Automatically downloads and manages ChromeDriver
service = Service(ChromeDriverManager().install())
chromeOptions = webdriver.ChromeOptions()
chromeOptions.timeouts = {"implicit": 10000}
chromeOptions.page_load_strategy = 'normal'

driver = webdriver.Chrome(service=service, options=chromeOptions)


pagesCount = 400

if (len(cartLinks) == 0):
    for f in range(1, pagesCount):
        pageLink = f"{linkP1}{f}{linkP2}"
        driver.get(pageLink)
        print(f"Get links from Page: {f}")

        xpath_selector = "//a"

        #linkTags = soup.find_all("a", attrs={"href": re.compile("/services/responses/bank/response/\d{8}/$")})

        linkTags = driver.find_elements(By.XPATH, xpath_selector)
        regex = re.compile("https://www.banki.ru/services/responses/bank/response/\d{8}/$")
        links = set()
        for link in linkTags:
            href = link.get_attribute("href")
            if (href and regex.match(href)):
                links.add(href)


        cartLinks += [l for l in links]
        sleep(random.uniform(1.0, 2.5))

    print(f"Dataset len: {len(cartLinks)} links")

    with open("links.txt", 'w') as file:
        for link in cartLinks:
            file.write(link)
            file.write("\n")

fieldNames = ["id", "time", "estimation", "text"]
data = []

for id, link in enumerate(cartLinks):
    driver.get(link)
    print(f"parsing {id} link...", end="")

    time = 0
    text = 0
    estimation = 0
    # '''<div font-size="fs18" class="MarkdownInsidestyled__MarkdownInsideStyled-sc-1frtivc-0 bKVLHc"><p>Решил принять участие в акции "пригласи друга". Пригласил, человек выполнил условия, потратил три тысячи - вознаграждение не начислили. В чате приложения не отвечают, а если отвечают, то быстро и не по делу, отключаются сразу выходят из чата. Когда спросил "на что нужно потратить три тысячи по условиям акции" , мне сказали "не владеем такой информацией". Это как? В банке сотрудники сами не знают, как выполнить условия акции "приведи друга"?</p></div>'''
#    lxmlTree = etree.HTML(response.text)
    try:
        textDivTag = driver.find_element(By.XPATH,
        "/html/body/div[3]/main/div[1]/section[1]/main/div/section/div[2]/div[2]/div/div[1]/div[2]/div")
    except:
        print(" server 500+ error. trying to reload...", end='')
        retryNum = 5
        retryDelay = 5
        for i in range(retryNum):
            print(f"{i}..", end="")
            sleep(retryDelay)
            driver.refresh()

            try:
                textDivTag = driver.find_element(By.XPATH,
"/html/body/div[3]/main/div[1]/section[1]/main/div/section/div[2]/div[2]/div/div[1]/div[2]/div")
                break
            except:
                pass

    pTag = None


    try:
        pTag = textDivTag.find_element(By.XPATH, ".//p")
        text = pTag.text.strip()
    except:
        text = textDivTag.text.strip()

    time = driver.find_element(By.XPATH,"/html/body/div[3]/main/div[1]/section[1]/header/div/div/div[2]/div/div/span").text.strip()

    estimation = driver.find_element(By.XPATH,"/html/body/div[3]/main/div[1]/section[1]/main/div/section/div[1]/div[1]/div/div[2]").text.strip()

    dataItem = {"id": id, "time": time, "estimation": estimation, "text": text}

    data.append(dataItem)
    print("Done!")
    sleep(random.uniform(1.0, 8.5))
driver.quit()

if len(data) > 0:
    with open('dataset.csv', 'w', errors='ignore', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(data)
    print("Parse is done!")

