import numpy as np
import math

def main():
    
    # задание 1
    test_1 = np.array([92, 94, 88, 91, 87])
    
    # задание 2
    test_2 = np.genfromtxt("test_2.csv", delimiter=",")
    print(test_2)
    
    # задание 3
    
    test_3 = np.array([87, 85, 72, 90, 92])
    test_3_fixed = (test_3 + 2)
    print(test_3_fixed)
    
    # задание 4
    
    total_grade = np.concatenate((test_1, test_2, test_3_fixed))
    final_grade = np.mean(total_grade)
    print(total_grade)
    print(final_grade)
    
    # задание 5 
    
    coin_toss = np.array([1, 0, 0, 1, 0])
    coin_toss_2 = np.array([0, 0, 1, 1, 1])
    
    coin_toss_again = np.vstack((coin_toss, coin_toss_2))
    print(coin_toss_again)
    
    # задание 6 
    jeremy_test_2 = test_2[3]
    manual_adwoa_test_1 = test_1[1:3]
    print(jeremy_test_2)
    print(manual_adwoa_test_1)
    
    # задание 7 
    
    student_scores = np.vstack((test_1, test_2, test_3_fixed))
    tanya_test_3 = student_scores[2, 0]
    print(student_scores)
    print(tanya_test_3)
        
    cody_test_scores = (lambda x: x[:, 4])(student_scores)
    print(cody_test_scores)
    
    # задание 8 

    temperature = np.genfromtxt("temperature_data.csv", delimiter=",")
    print(f"original temp: {temperature}")
    temperature_fixed = temperature + 3
    print(f"temp plus 3 {temperature_fixed}")
    monday_temperas = temperature_fixed[0]
    print(f"monday temperature: {monday_temperas}")
    thursday_friday_morning = temperature_fixed[3:5, 1]
    print(f"6 am temp on thu and fri: {thursday_friday_morning}\n")
    temperature_extremes = np.where(
        (temperature_fixed < 50) | (temperature_fixed > 60), 
        temperature_fixed, 
        np.nan
    )
    print(f"temperature extremes: {temperature_extremes}")
    
    # задание 9 
    
    spiral()
    
    
    # задание 10
    
    snake()

    
def spiral():
    
    import matplotlib.pyplot as plt
    
    n = np.arange(97)

    phi = n * math.pi / 12
    

    x = phi * np.cos(phi)
    y = phi * np.sin(phi)

    table = np.array([n, phi, x, y]).T

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='.', color='teal', linestyle='-', label='спираль')

    plt.show()
    
def snake():
    import matplotlib.pyplot as plt
    import numpy as np

    n = np.arange(24)
    phi = np.pi * n / 12

    x = 2 * np.cos(phi)**2 + np.cos(phi)
    y = 2 * np.sin(phi) * np.cos(phi) + np.sin(phi)

    table = np.array([n, phi, x, y]).T

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='.', color='teal', linestyle='-', label='улитка')

    plt.show()


if __name__ == "__main__":
    main()
