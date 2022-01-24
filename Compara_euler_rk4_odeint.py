# Importando bibliotecas necessárias para o projeto.
import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Definindo tema da interface gráfica.
sg.theme('Default1')  

# Criando layout da interface gráfica.    
layout = [  
            [sg.Text('Modelagem SEIRS:')],
            [sg.Text('Tamanho da população:')],
            [sg.Slider(range=(0,2e9), default_value=2e8, resolution=10000,
            size=(50,15), orientation='horizontal', font=('Helvetica', 12),key='-popsize-')],
            [sg.Text('UTIs disponíveis(por 10000 habitantes):')],
            [sg.Slider(range=(0,20), default_value=1, resolution=0.1,
            size=(50,15), orientation='horizontal', font=('Helvetica', 12),key='-uti-')],
            [sg.Text('Taxa de internação nas UTIs:')],
            [sg.Slider(range=(0,1), default_value=0.05, resolution=0.01,
            size=(50,15), orientation='horizontal', font=('Helvetica', 12),key='-internacao-')],
            [sg.Text('R0 (Número de Reprodução Básica):')],
            [sg.Slider(range=(0,10), default_value=2.5, resolution=0.01,
            size=(50,15), orientation='horizontal', font=('Helvetica', 12),key='-repr-')],
            [sg.Text('Tempo(anos):')],
            [sg.Slider(range=(0,10), default_value=2, resolution=0.1,
            size=(50,15), orientation='horizontal', font=('Helvetica', 12),key='-time-')],
            [sg.Text('Nível de distanciamento social:',)],
            [sg.Slider(range=(0,1), default_value=0.2, resolution=0.01,
            size=(50,15), orientation='horizontal', font=('Helvetica', 12),key='-distance-')],
            [sg.Text('Parâmetros:')],
            [sg.Text('Período de Incubação(dias): '), sg.Slider(range=(0,10), default_value=5.1, resolution=0.1,
            size=(10,15), orientation='horizontal', font=('Helvetica',12), key= '-incubacao-')], 
            [sg.Text('Período de infecção(dias): '),  sg.Slider(range=(0,10), default_value=3.3, resolution=0.1,
            size=(10,15), orientation='horizontal', font=('Helvetica',12), key= '-infeccao-')],
            [sg.Text('Período de imunidade(dias): '),  sg.Slider(range=(0,730), default_value=365, resolution=1,
            size=(10,15), orientation='horizontal', font=('Helvetica',12), key= '-imunidade-')],
            [sg.Button('Ok'), sg.Button('Cancel')]]
    

# Criando a janela.
window = sg.Window('Modelo SEIRS COVID-19', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # Fecha janela se clicar no ícone de fechar ou no botão de cancelar.
        break
    if event == 'Ok':
        # Função que define as EDO´s da modelagem SEIRS para uma epidemia.
        def SEIRS_MODEL(x, t, params, N):
            alpha = params["Alpha"]
            beta = params["Beta"]
            gamma = params["Gamma"]
            omega = params["Omega"]
            mi = params["Mi"]
            SEIRSdot = np.zeros(4)
            SEIRSdot[0] = -(1-mi)*(beta*x[0]*x[2]/N) + omega*x[3] #dS/dt
            SEIRSdot[1] =  (1-mi)*(beta*x[0]*x[2]/N) - alpha*x[1] #dE/dt
            SEIRSdot[2] =   alpha*x[1] - gamma*x[2] #dI/dt
            SEIRSdot[3] = gamma*x[2] - omega*x[3] #dR/dt
            return SEIRSdot


        # Método euler para computar a evolução das EDO´s ao longo do tempo.
        def euler(f, x0, t0, tf, dt):
            t = np.arange(t0,tf,dt)
            nt = t.size
            nx = x0.size
            x = np.zeros([nx,nt])
            
            x[:,0] = x0

            for k in range(nt-1):
                x[:,k+1] = (x[:,k] + dt*f(t[k],x[:,k]))

            return x, t

        # Método Runge-Kutta(Quarta Ordem) para computar a evolução das EDO´s ao longo do tempo.
        def RK4(f, x0, t0, tf, dt):
            t = np.arange(t0,tf,dt)
            nt = t.size
            nx = x0.size
            x = np.zeros([nx,nt])
            
            x[:,0] = x0

            for k in range(nt-1):

                k1 = dt*f(t[k],x[:,k])
                k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2)
                k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2)
                k4 = dt*f(t[k] + dt, x[:,k] + k3)

                dx = (k1+2*k2+2*k3+k4)/6
                
                x[:,k+1] = x[:,k] + dx
          
            return x, t

        # Parâmtros de tempo(dias).
        t_incubacao = float(values['-incubacao-']) # 5.1
        t_infeccao = float(values['-infeccao-']) # 3.3
        t_imunidade = float(values['-imunidade-']) # 365

        # Número de Reprodução Básica.
        R0 = float(values['-repr-']) # 2.5

        # População
        N = int(values['-popsize-']) # 200000000

        # Nivel de distanciamento social.
        # 0.0 - Interação social sem restrições;
        # 0.7 - Lockdown(na prática);
        # 1.0 - Isolamento total(ideal).
        u = float(values['-distance-'])

        # Parâmetros da modelagem SEIRS - COVID-19.
        params = {'R0': R0, 'Alpha': 1/t_incubacao, 'Beta': R0*1/t_infeccao,'Gamma':1/(t_infeccao), 'Omega':1/t_imunidade,'Mi': u}

        # Cálculo de Runge-Kutta.
        f = lambda t, x : SEIRS_MODEL(x, t, params, N)

        # Condições iniciais do modelo.
        e0 = 1
        i0 = 0
        r0 = 0
        s0 = N - e0 -i0 - r0
        SEIRS_0 = np.array([s0,e0,i0,r0])

        # Tempo de simulação e passo.
        t0 = 0
        tf = 365*int(values['-time-'])
        dt = 0.01

        # Recebendo arrays resultados dos métodos de Euler, Runge-Kutta(quarta ordem) e ODEINT.
        x_euler,t=  euler(f, SEIRS_0, t0, tf, dt)
        x_rk4,t =  RK4(f, SEIRS_0, t0, tf, dt)
        x_odeint = odeint(SEIRS_MODEL,SEIRS_0, t, args = (params, N))

        symbol = ['S','E','I','R']
        print("Passo utilizado(h):", dt)
        print('\n')

        # Calculando o erro quadrático médio de RK4 utilizando ODEINT como valor de referência.
        eqm_RK4 = []
        for i in range(0, 4):
            eqm_RK4.append(((x_rk4[i,:] - x_odeint[:,i])**2).mean(axis=None))
            txt = "Erro quadrático médio {} - RK4 e ODEINT: ".format(symbol[i])
            print(txt, eqm_RK4[i])
        print()
        
        # Calculando o erro quadrático médio de euler utilizando ODEINT como valor de referência.
        eqm_euler = []
        for i in range(0, 4):
            eqm_euler.append(((x_euler[i,:] - x_odeint[:,i])**2).mean(axis=None))
            txt = "Erro quadrático médio {} - Euler e ODEINT: ".format(symbol[i])
            print(txt,eqm_euler[i])


        # Plotagem dos gráficos.
        fig, ax = plt.subplots(3, 1)

        # Gráfico de Euler
        ax[0].set_title("Simulação epidemiológica SEIRS - Euler")
        ax[0].plot(t/365, x_euler[0,:], 'r', label = 'S')
        ax[0].plot(t/365, x_euler[1,:], 'g', label = 'E')
        ax[0].plot(t/365, x_euler[2,:], 'b', label = 'I')
        ax[0].plot(t/365, x_euler[3,:], 'y', label = 'R')
        ax[0].set_ylabel('População')
        ax[0].set_xlabel('tempo(anos)')
        ax[0].grid()
        ax[0].legend()

        # Gráfico de Runge-Kutta(Quarta Ordem).
        ax[1].set_title("Simulação epidemiológica SEIRS - RK4")
        ax[1].plot(t/365, x_rk4[0,:], 'r', label = 'S')
        ax[1].plot(t/365, x_rk4[1,:], 'g', label = 'E')
        ax[1].plot(t/365, x_rk4[2,:], 'b', label = 'I')
        ax[1].plot(t/365, x_rk4[3,:], 'y', label = 'R')
        ax[1].set_ylabel('População')
        ax[1].set_xlabel('tempo(anos)')
        ax[1].grid()
        ax[1].legend()

        # Gráfico de ODEINT. 
        ax[2].set_title("Simulação epidemiológica SEIRS - ODEINT")
        ax[2].plot(t/365, x_odeint[:,0], 'r', label = 'S')
        ax[2].plot(t/365, x_odeint[:,1], 'g', label = 'E')
        ax[2].plot(t/365, x_odeint[:,2], 'b', label = 'I')
        ax[2].plot(t/365, x_odeint[:,3], 'y', label = 'R')
        ax[2].set_ylabel('População')
        ax[2].grid()
        ax[2].legend()
        ax[2].set_xlabel('tempo(anos)')

        plt.subplots_adjust(hspace=0.8)
        plt.show()