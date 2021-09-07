import numpy as np

labels = True

metrics = np.load("SSMnet_Beatles_labels.npy")

print("F3 : ", np.round(np.mean(metrics[0]),3), "+/-", np.round(np.std(metrics[0]),3))
print("P3 : ", np.round(np.mean(metrics[1]),3), "+/-", np.round(np.std(metrics[1]),3))
print("R3 : ", np.round(np.mean(metrics[2]),3), "+/-", np.round(np.std(metrics[2]),3))
print("F0.5 : ", np.round(np.mean(metrics[3]),3), "+/-", np.round(np.std(metrics[3]),3))
print("P0.5 : ", np.round(np.mean(metrics[4]),3), "+/-", np.round(np.std(metrics[4]),3))
print("R0.5 : ", np.round(np.mean(metrics[5]),3), "+/-", np.round(np.std(metrics[5]),3))
print("F3w : ", np.round(np.mean(metrics[6]),3), "+/-", np.round(np.std(metrics[6]),3))
print("F0.5w : ", np.round(np.mean(metrics[7]),3), "+/-", np.round(np.std(metrics[7]),3))

if labels:
    print("PWF : ", np.round(np.mean(metrics[8]),3), "+/-", np.round(np.std(metrics[8]),3))
    print("PWP : ", np.round(np.mean(metrics[9]),3), "+/-", np.round(np.std(metrics[9]),3))
    print("PWR : ", np.round(np.mean(metrics[10]),3), "+/-", np.round(np.std(metrics[10]),3))
    print("Sf : ", np.round(np.mean(metrics[11]),3), "+/-", np.round(np.std(metrics[11]),3))
    print("So : ", np.round(np.mean(metrics[12]),3), "+/-", np.round(np.std(metrics[12]),3))
    print("Su : ", np.round(np.mean(metrics[13]),3), "+/-", np.round(np.std(metrics[13]),3))
    