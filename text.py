import cv2

#图片存在的实际路径
input_path = r"N:\Pythonproject\python_base_study\test.jpg"

# 修改为希望保存在哪里
output_path = r"N:\Pythonproject\total\test.jpg"

# 读取图片
image = cv2.imread(input_path)

# 判断图片是否读取成功
if image is None:
    print("图片读取失败，请检查图片路径和文件名是否正确。")
    exit()

print("图片读取成功！")

# 显示图片
cv2.imshow("Image", image)

# 等待按键
cv2.waitKey(0)

# 关闭显示窗口
cv2.destroyAllWindows()

# 保存图片
if cv2.imwrite(output_path, image):
    print("图片保存成功，保存路径为：")
    print(output_path)
else:
    print("图片保存失败，请检查保存路径是否正确。")