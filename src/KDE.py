
#########################################################
###7#####compute KDE ##
####################################################
X = gym.head_posx.values
Y = gym.head_posy.values
Z = gym.head_posz.values

xmin = X.min()
xmax = X.max()
ymin = Y.min()
ymax = Y.max()
zmin = Z.min()
zmax = Z.max()




values = np.vstack([X, Y, Z])

density = stats.gaussian_kde(values)(values)
Zdensity = (density - density.min())/(density.max()-density.min())*100
#gym["density"]=density
#gym["zdensity"]=Zdensity



idx = Zdensity.argsort()
X, Y, Z, Zdensity = X[idx], Y[idx], Z[idx], Zdensity[idx]