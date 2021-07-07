from selenium import webdriver
from time import sleep


url = "https://www.amazon.es/s?k=laptop&__mk_es_ES=%C3%85M%C3%85%C5%BD%C3%95%C3%91&ref=nb_sb_noss_1"
my_driver = "chromedriver.exe"

driver = webdriver.Chrome(my_driver)
webdriver.c

driver.get(url)

sleep(3)

#class="a-price-whole"

laptop_price = driver.find_element_by_xpath('//a[@class="a-link-normal a-text-normal"]')

laptop_price.click()

sleep(1)
buy_zone = driver.find_element_by_xpath('//div[@class="a-box-inner"]')


id="price_inside_buybox"
buy_price = buy_zone.find_element_by_xpath('//span[@id="price_inside_buybox"]')

print(buy_price.text)
