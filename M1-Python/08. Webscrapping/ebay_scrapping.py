from selenium import webdriver
import selenium
from selenium.webdriver.common.keys import Keys
from time import sleep


url = 'https://www.ebay.com/'
my_driver = "chromedriver.exe"

driver = webdriver.Chrome(my_driver)

driver.get(url)

sleep(1)

search_bar = driver.find_element_by_xpath('/html/body/header/table/tbody/tr/td[5]/form/table/tbody/tr/td[1]/div[1]/div/input[1]')

search_bar.send_keys("laptop")

#                               tag   @ atribute
search_button =  driver.find_element_by_xpath('//input[@class="btn btn-prim gh-spr"]')

search_button.click()

sleep(3)





screen_size =  driver.find_element_by_xpath('/html/body/div[4]/div[3]/ul/li[1]/ul/li[2]/ul/li[1]/div[2]/div/div/button')
screen_size.click()


sleep(3)



os =  driver.find_element_by_xpath('/html/body/div[13]/div[2]/div/form/div[1]/div[2]/div[1]/div/fieldset/div[1]/div[5]/label/div/span/input')
os.click()

sleep(1)



screen =  driver.find_element_by_xpath('/html/body/div[13]/div[2]/div/form/div[1]/div[1]/div/div[3]')
screen .click()

sleep(2)



screen_size =  driver.find_element_by_xpath('/html/body/div[13]/div[2]/div/form/div[1]/div[2]/div[1]/div/fieldset/div[1]/div[2]/label/div/span/input')
screen_size.click()

sleep(1)



apply =  driver.find_element_by_xpath('/html/body/div[13]/div[2]/div/form/div[3]/div[2]/button')
apply.click()