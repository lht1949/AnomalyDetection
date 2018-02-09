import numpy as np
import pywt

def solve_theta(decomp, mu_1):
    eigval, eigvec = np.linalg.eigh(decomp)
    Q = np.matrix(eigvec)
    D = eigval + np.sqrt(np.square(eigval)+ 4*mu_1)
    xdiag = np.matrix(np.diag(D))
    return 1./(2*mu_1)*Q*xdiag*Q.T

def solve_L(theta_, M, S, mu_2, U_2):
    return M - S + 1./mu_2*(U_2 - theta_)


def projectSPD(A):
    eigval, eigvec = np.linalg.eigh(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T


def Robust_glasso_S_first(M, pho_, lambda_, maxIteration=1000, epsilon=None):

    if epsilon is None:
        epsilon = 1e-7

    U_1 = M*0.0
    U_2 = M*0.0
    Z = M*0.0
    theta_ = M*0.0
    theta_temp = M*0.0
    L = M*0.0
    S = M - L

    mu_1 = 0.2
    mu_2 = 0.2

    iteration = 0

    while True:
        # Break if we use too many interations
        iteration += 1
        if iteration > maxIteration:
            break


        # step4 soft thresholding S
        threshold_S = lambda_*1./mu_2
        soft_target_S = M - L + (1./mu_2)*U_2


        #print 'soft_target_S',  soft_target_S
        #print "threshold_S", threshold_S
        S = pywt.threshold(soft_target_S, threshold_S, 'soft')
        S = np.matrix(S)
        #print 'after soft', S

        # step3 sovle and project L
        L_temp = solve_L(theta_, M, S, mu_2, U_2)
        L = projectSPD(L_temp)



        # update U_2
        U_2 = U_2 + mu_2*(M - L - S)


        # step1 solve theta
        decomp = mu_1*(Z - U_1) - L
        theta_ = solve_theta(decomp, mu_1)

        # step2 soft thresholding Z
        threshold_Z = pho_*1./mu_1
        soft_target_Z = theta_ + U_1


        Z = pywt.threshold(soft_target_Z, threshold_Z, 'soft')
        Z = np.matrix(Z)

        # update U_1
        U_1 = U_1 + theta_ - Z


        if iteration > 1:
            criterion1 = (np.linalg.norm((theta_ - theta_temp ), 'fro')**2)/(np.linalg.norm(theta_temp, 'fro')**2)
            criterion2 = (np.linalg.norm((M - L - S), 'fro')**2)/(np.linalg.norm(M, 'fro')**2)

            if criterion1 < epsilon and criterion2 < epsilon:
                break

        mu_1 *= 1.2
        mu_2 *= 1.2
        theta_temp = theta_.copy()


    return theta_, S, L, iteration, criterion1, criterion2
