# Importando bibliotecas necessárias para o projeto.
import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt

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
            [sg.Slider(range=(0,10), default_value=2, resolution=1,
            size=(50,15), orientation='horizontal', font=('Helvetica', 12),key='-time-')],
            [sg.Text('Nível de distanciamento social:',)],
            [sg.Slider(range=(0,1), default_value=0.2, resolution=0.01,
            size=(50,15), orientation='horizontal', font=('Helvetica', 12),key='-distance-')],
            [sg.Text('Lockdown de emergência:'), sg.Spin(values=('No', 'Yes'), initial_value='No',size=(5,15),
             font=('Helvetica', 12),key='-lockdown-')],
            [sg.Text('Parâmetros:')],
            [sg.Text('Período de Incubação(dias): '), sg.Slider(range=(0,10), default_value=5.1, resolution=0.1,
            size=(10,15), orientation='horizontal', font=('Helvetica', 12), key= '-incubacao-')], 
            [sg.Text('Período de infecção(dias): '), sg.Slider(range=(0,10), default_value=3.3, resolution=0.1,
            size=(10,15), orientation='horizontal', font=('Helvetica', 12), key= '-infeccao-')],
            [sg.Text('Período de imunidade(dias): '), sg.Slider(range=(0,730), default_value=365, resolution=1,
            size=(10,15), orientation='horizontal', font=('Helvetica', 12), key= '-imunidade-')],
            [sg.Button('Ok'), sg.Button('Cancel')]]
    

# Criando a janela.
window = sg.Window('Modelo SEIRS COVID-19', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # Fecha janela se clicar no ícone de fechar ou no botão de cancelar.
        break
    if event == 'Ok':

        # Função que define as EDO´s da modelagem SEIRS para uma epidemia.
        def SEIRS_MODEL(x, params, N, u):
            # Coleta de parâmetros
            alpha = params["Alpha"]
            beta = params["Beta"]
            gamma = params["Gamma"]
            omega = params["Omega"]
            mi = params["Mi"]

            # Decisão entre parâmetro Mi inicial ou parâmetro Mi de Lockdown
            amort = u if u != mi else mi

            # Array com Edos do modelo.
            SEIRSdot = np.array([-(1-amort)*(beta*x[0]*x[2]/N) + omega*x[3] , #dS/dt
                            (1-amort)*(beta*x[0]*x[2]/N) - alpha*x[1], #dE/dt
                            alpha*x[1] - gamma*x[2], #dI/dt
                            gamma*x[2] - omega*x[3] #dR/dt
                            ]) 
            return SEIRSdot


        # Método Runge-Kutta (Quarta Ordem) para computar a evolução das EDO´s ao longo do tempo.
        def RK4_lockdown(f, x0, t0, tf, dt, params,N):
            t = np.arange(t0,tf,dt)
            nt = t.size
            nx = x0.size
            x = np.zeros([nx,nt])
            x[:,0] = x0
             
            # Em média, 1 a cada 20 pessoas infectadas com a COVID-19 necessita de uma UTI.
            # O Brasil, durante os piores momentos da pandemia, disponibizou 1 UTI para cada 10000 habitantes do país.
            # Para calcular a capacidade das UTIs ao longo do tempo.
            ICU = 1/float(values['-internacao-'])*float(values['-uti-'])/10000*N
            icu = [ICU]*nt

            # Para criar o vetor com dados da variação do número de reprodução básica ao longo do tempo.
            r0 = params["R0"]
            rt = [r0]

            # Para calcular a porcentagem de transmisoes que se deve reduzir para controlar uma epidemia.
            reduc_transm = 1-(1/r0)

            mi = params["Mi"]

            k = 0
            mes = 30
            while k < nt-1:
                # Condições para um lockdown de emergência seja acionado.
                if mi < reduc_transm and x[2,k] > ICU and values['-lockdown-'] == 'Yes':
                    count = 1
                    # Um Lockdown de emergência dura 1 mês nessa simulação. 
                    while count < mes/dt:
                        if  k == nt-1:
                            break
                        mi = 0.7
                        k1 = dt*f(t[k],x[:,k], mi)
                        k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2, mi)
                        k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2, mi)
                        k4 = dt*f(t[k] + dt, x[:,k] + k3, mi)

                        dx = (k1+2*k2+2*k3+k4)/6
                
                        x[:,k+1] = x[:,k] + dx
                        
                        # Cálculo da variação do número básico de reprodução ao longo do tempo.
                        rt.append(r0*((1-mi)*x[0,k]/N))
                        count += 1
                        k += 1
                else:       
                    mi = params["Mi"]

                    k1 = dt*f(t[k],x[:,k], mi)
                    k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2, mi)
                    k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2, mi)
                    k4 = dt*f(t[k] + dt, x[:,k] + k3, mi)

                    dx = (k1+2*k2+2*k3+k4)/6

                    x[:,k+1] = x[:,k] + dx

                    # Cálculo da variação do número de reprodução ao longo do tempo.
                    rt.append(r0*((1-mi)*x[0,k]/N))

                    k += 1
        
            return x, t, rt, icu


        # Parâmetros de tempo(dias)
        t_incubacao = float(values['-incubacao-']) # 5.1
        t_infeccao = float(values['-infeccao-']) # 3.3
        t_imunidade = float(values['-imunidade-']) # 365

        # Número de Reprodução Básica.
        R0 = float(values['-repr-']) # 2.5

        # População
        N = int(values['-popsize-']) # 20000000

        # Nivel de distanciamento social.
        # 0.0 - Interação social sem restrições;
        # 0.7 - Lockdown(na prática);
        # 1.0 - Isolamento total(ideal).
        u = float(values['-distance-']) # 0.2

        # Parâmetros da modelagem SEIRS.
        params = {'R0': R0, 'Alpha': 1/t_incubacao, 'Beta': R0*1/t_infeccao,'Gamma':1/(t_infeccao), 'Omega':1/t_imunidade,'Mi': u}

        
        f = lambda t, x, u : SEIRS_MODEL(x, params, N, u)

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


        # Cálculo de Runge-Kutta.
        x,t,rt,icu =  RK4_lockdown(f, SEIRS_0, t0, tf, dt, params,N)


        # Plotando dos gráficos.
        fig, ax = plt.subplots(2, 1)
        
        # Gráfico da simulação epidemiológica SEIRS.
        ax[0].set_title('Simulação epidemiológica SEIRS')
        ax[0].plot(t/365, x[0,:], 'r', label = 'S')
        ax[0].plot(t/365, x[1,:], 'g', label = 'E')
        ax[0].plot(t/365, x[2,:], 'b', label = 'I')
        ax[0].plot(t/365, x[3,:], 'y', label = 'R')
        ax[0].plot(t/365, icu, linestyle = '--', color = 'k', label = 'Capacidade das UTIs')
        ax[0].set_xlabel('tempo(anos)')
        ax[0].set_ylabel('População')
        ax[0].grid()
        ax[0].legend()
        
        # Gráfico da variação do número de reprodução básica ao longo do tempo.
        ax[1].set_title('Número de Reprodução')
        ax[1].plot(t/365, rt, label = 'RT')
        ax[1].set_xlabel('tempo(anos)')
        ax[1].grid()
        ax[1].legend()
        

        plt.subplots_adjust(hspace=0.8)
        plt.show()

window.close()