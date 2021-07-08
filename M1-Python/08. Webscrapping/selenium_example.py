from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
##############READ TO USE AN ADBLOCKER

page = "https://forecast.weather.gov/MapClick.php?lat=37.777120000000025&lon=-122.41963999999996"
my_driver = ("./chromedriver.exe")


driver = webdriver.Chrome(my_driver)
driver.create_options()
driver.minimize_window()
driver.get(page)
sleep(3)

san_fran = driver.find_element_by_xpath('//h2[@class="a-color-base headline truncate-2line"]')
print(san_fran.text)
'''
san_fran = driver.find_element_by_xpath('//div[@class="panel-body"]')

temp = san_fran.find_elements_by_xpath('//p[@class="temp temp-high"]')

for t in temp:
    print(t.text)
'''