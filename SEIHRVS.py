# Importando bibliotecas necessárias para o projeto.
import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt

# Definindo tema da interface gráfica.
sg.theme('Default1')  

# Criando layout da interface gráfica.   
font_size = 7
font_style = 'Helvetica'
left_col = [  
            [sg.Text('Tamanho da população:')],
            [sg.Slider(range=(0,8e9), default_value=2e8, resolution=10000,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-popsize-')],
            [sg.Text('Aumento Populacional (Nascimentos + Saldo imigratório):')],
            [sg.Slider(range=(0,2e6), default_value=10000, resolution=1,
            size=(50,10), orientation='horizontal', font=(font_style, font_size), key= '-aumentoPopulacao-')], 
            [sg.Slider(range=(0,20), default_value=1, resolution=0.1,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-uti-')],
            [sg.Text('Taxa de internação nas UTIs:')],
            [sg.Slider(range=(0,1), default_value=0.05, resolution=0.01,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-internacao-')],
            [sg.Text('R0 (Número de Reprodução Básica):')],
            [sg.Slider(range=(0,20), default_value=2.5, resolution=0.1,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-repr-')],
            [sg.Text('Tempo(anos):')],
            [sg.Slider(range=(0,10), default_value=5, resolution=1,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-time-')],
            [sg.Text('Nível de distanciamento social:')],
            [sg.Slider(range=(0,1), default_value=0.2, resolution=0.01,
            size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-distance-')],
            [sg.Text('Período de Incubação(dias): '), sg.Slider(range=(0,10), default_value=5.1, resolution=0.1,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-incubacao-')], 
            [sg.Text('Período de infecção(dias): '), sg.Slider(range=(0,10), default_value=3.3, resolution=0.1,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-infeccao-')],
            [sg.Text('Período de imunidade(dias) - Recuperados: '), sg.Slider(range=(0,730), default_value=365, resolution=1,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-imunidadeNatural-')],
            [sg.Text('Período de imunidade(dias) -  Vacinados: '), sg.Slider(range=(0,730), default_value=365, resolution=1,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-imunidadeVacinados-')],
            [sg.Text('Taxa de mortalidade natural(dias): '), sg.Slider(range=(0,1), default_value=0.3, resolution=0.01,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-mortalidadeNatural-')],
            [sg.Text('Taxa de mortalidade de Infectados(dias): '), sg.Slider(range=(0,1), default_value=0.3, resolution=0.01,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-mortalidadeInfectados-')],
            [sg.Text('Taxa de mortalidade de Hospitalizados(dias): '), sg.Slider(range=(0,1), default_value=0.3, resolution=0.01,
            size=(10,10), orientation='horizontal', font=(font_style, font_size), key= '-mortalidadeHospitalizados-')],
]

middle_col = [ 
    [sg.Text('Condições inciais:')],
    [sg.Text('Pessoas Expostas:')],
    [sg.Slider(range=(0,2e9), default_value=0, resolution=1000,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-E0-')],
    [sg.Text('Pessoas Infectadas:')],
    [sg.Slider(range=(0,2e9), default_value=0, resolution=1000,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-I0-')],
    [sg.Text('Pessoas Hospitalizadas:')],
    [sg.Slider(range=(0,2e9), default_value=0, resolution=1000,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-H0-')],
    [sg.Text('Pessoas Vacinadas:')],
    [sg.Slider(range=(0,2e9), default_value=0, resolution=1000,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-V0-')],
    [sg.Text('Pessoas Recuperadas:')],
    [sg.Slider(range=(0,2e9), default_value=0, resolution=1000,
    size=(50,10), orientation='horizontal', font=(font_style, font_size),key='-R0-')],
]

right_col = [
    [sg.Text('Ações de Controle: ')],
    [sg.Text('Vacinação: '), sg.Spin(values=('No', 'Yes'), initial_value='No',size=(5,10),
    font=(font_style, font_size),key='-vacinacao-')],
    [sg.Text('Taxa de vacinação por dia: '), sg.Slider(range=(0,1), default_value=0.0021, resolution=0.0001,
    size=(10,10), orientation='horizontal', font=(font_style, font_size), key='-taxa-vacinacao-')],
    [sg.Text('Taxa de efetividade da vacinação: '), sg.Slider(range=(0,1), default_value=0.8, resolution=0.01,
    size=(10,10), orientation='horizontal', font=(font_style, font_size), key='-taxa-efetividade-')],
    [sg.Text('Tempo para início da vacinação: ')],
    [sg.Slider(range=(0,10), default_value=2, resolution=1,
    size=(50,10), orientation='horizontal', font=(font_style, font_size), key='-tempo-vacinacao-')],
    [sg.Text('Lockdown de emergência:'), sg.Spin(values=('No', 'Yes'), initial_value='No',size=(5,10),
     font=(font_style, 8),key='-lockdown-')]
]
           
layout = [
    [
        [sg.Text('Modelagem SEIHRVS - Parâmetros:')],
        sg.Column(left_col, vertical_alignment='top'), 
        sg.Column(middle_col, vertical_alignment='top'),
        sg.Column(right_col, vertical_alignment='top'),
    ],
    [sg.Button('Ok'), sg.Button('Cancel')]
]

    

# Criando a janela.
window = sg.Window('Modelo SEIHRVS COVID-19', layout, finalize=True)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Cancel': # Fecha janela se clicar no ícone de fechar ou no botão de cancelar.
        break
    if event == 'Ok':

        # Função que define as EDO's da modelagem SEIHRVS para uma epidemia.
        def SEIHRVS_MODEL(x, params, N, u, t, ICU):
            # Coleta de parâmetros
            alpha = params["Alpha"]
            beta = params["Beta"]
            gammaI = params["GammaI"]
            gammaH = params["GammaH"]
            delta = params["Delta"]
            muI = params["MuI"]
            muH = params["MuH"]
            omegaR = params["OmegaR"]
            omegaV = params["OmegaV"]
            pi = params["pi"]
            epsilon = params["epsilon"]
            e = params["e"] if params["VacinaAtiva"] and t >= params["TempoInicioVacinacao"] else 0.0
            v = params["v"] if params["VacinaAtiva"] and t >= params["TempoInicioVacinacao"] else 0.0

            tau = params["tau"]

            # Decisão entre parâmetro tau inicial ou parâmetro tau de Lockdown
            amort = u if u != tau else tau

            vacina_ativa = params["VacinaAtiva"] and t >= params["TempoInicioVacinacao"]
            fluxo_vacinacao = e*v*x[0] - (pi + omegaV)*x[5] if vacina_ativa else 0.0 
            sobrecapacidade_uti = x[3] > ICU

            # Array com Edos do modelo.
            SEIHRVSdot = np.array([-(1-amort)*(beta*x[0]*x[2]/N) + omegaR*x[4] - pi*x[0] - fluxo_vacinacao + epsilon, #dS/dt
                            (1-amort)*(beta*x[0]*x[2]/N) - (alpha + pi)*x[1], #dE/dt
                            alpha*x[1] - (gammaI + delta + pi + muI)*x[2] , #dI/dt
                            delta*x[2] - (gammaH + muH + pi)*x[3] if not sobrecapacidade_uti else  -(gammaH + muH)*x[3], #dH/dt
                            gammaI*x[2] + gammaH*x[3] - (omegaR + pi)*x[4], #dR/dt
                            fluxo_vacinacao  #dV/dt
                            ]) 
            return SEIHRVSdot


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
            ICU = (float(values['-uti-'])/10000)*N
            icu = [ICU]*nt

            # Para criar o vetor com dados da variação do número de reprodução básica ao longo do tempo.
            r0 = params["R0"]
            rt = [r0]

            # Para calcular a porcentagem de transmisoes que se deve reduzir para controlar uma epidemia.
            tau = params["tau"]

            k = 0
            mes = 30
            while k < nt-1:
                # Condições para um lockdown de emergência seja acionado.
                if x[3,k] > 0.8*ICU and values['-lockdown-'] == 'Yes':
                    count = 1
                    # Um Lockdown de emergência dura 1 mês nessa simulação. 
                    while count < mes/dt:
                        if  k == nt-1:
                            break
                        tau = 0.7
                        k1 = dt*f(t[k],x[:,k], tau)
                        k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2, tau)
                        k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2, tau)
                        k4 = dt*f(t[k] + dt, x[:,k] + k3, tau)

                        dx = (k1+2*k2+2*k3+k4)/6
                
                        x[:,k+1] = x[:,k] + dx
                        
                        # Cálculo da variação do número básico de reprodução ao longo do tempo.
                        rt.append(r0*((1-tau)*x[0,k]/N))
                        count += 1
                        k += 1
                else:       
                    tau = params["tau"]

                    k1 = dt*f(t[k],x[:,k], tau)
                    k2 = dt*f(t[k] + dt/2, x[:,k] + k1/2, tau)
                    k3 = dt*f(t[k] + dt/2, x[:,k] + k2/2, tau)
                    k4 = dt*f(t[k] + dt, x[:,k] + k3, tau)

                    dx = (k1+2*k2+2*k3+k4)/6

                    x[:,k+1] = x[:,k] + dx

                    # Cálculo da variação do número de reprodução ao longo do tempo.
                    rt.append(r0*((1-tau)*x[0,k]/N))

                    k += 1
        
            return x, t, rt, icu


        # Parâmetros de tempo(dias)
        t_incubacao = float(values['-incubacao-']) # 5.1
        t_infeccao = float(values['-infeccao-']) # 3.3
        t_imunidade = float(values['-imunidade-']) # 365
        tx_internacao = float(values['-internacao-']) # 0.05
        vacinacao_ativa = values['-vacinacao-'] == 'Yes'
        tx_vacinacao = float(values['-taxa-vacinacao-']) if vacinacao_ativa else 0.0 # 0.2% da população é vacinada por dia
        tx_efetividade = float(values['-taxa-efetividade-']) if vacinacao_ativa else 0.0
        tempo_inicio_vacinacao_anos = float(values['-tempo-vacinacao-']) if vacinacao_ativa else 0.0
        tempo_inicio_vacinacao_dias = tempo_inicio_vacinacao_anos * 365
        tx_mortalidade = float(values['-mortalidade-']); #30% dos hospitalizados falecem

        # Número de Reprodução Básica.
        R0 = float(values['-repr-']) # 2.5

        # População
        N = int(values['-popsize-']) # 20000000

        # Nivel de distanciamento social.
        # 0.0 - Interação social sem restrições;
        # 0.7 - Lockdown(na prática);
        # 1.0 - Isolamento total(ideal).
        u = float(values['-distance-']) # 0.2

        # Parâmetros da modelagem SEIHRVS.
        params = {'R0': R0, 'Alpha': 1/t_incubacao, 'Beta': R0*1/t_infeccao,'GammaI':1/t_infeccao, 'Delta':tx_internacao, 'GammaH':(1-tx_mortalidade), 'Mu':tx_mortalidade, 'Omega':1/t_imunidade, 'v': tx_vacinacao, 'e': tx_efetividade,'tau': u, 'VacinaAtiva': vacinacao_ativa, 'TempoInicioVacinacao': tempo_inicio_vacinacao_dias}

        ICU = (float(values['-uti-'])/10000)*N
        f = lambda t, x, u : SEIHRVS_MODEL(x, params, N, u, t, ICU)

        # Condições iniciais do modelo.
        e0 = int(values['-E0-']);
        i0 = int(values['I0-']);
        h0 = int(values['-H0-']);
        r0 = int(values['-R0-']);
        v0 = int(values['-V0-']);
        s0 = N - e0 -i0 - r0 - h0 - v0
        SEIHRVS_0 = np.array([s0,e0,i0,h0,r0,v0])

        # Tempo de simulação e passo.
        t0 = 0
        tf = 365*int(values['-time-'])
        dt = 1

        # Cálculo de Runge-Kutta.
        x,t,rt,icu =  RK4_lockdown(f, SEIHRVS_0, t0, tf, dt, params,N)


        # Plotando dos gráficos.
        fig, ax = plt.subplots(2, 1)
        
        # Gráfico da simulação epidemiológica SEIHRVS.
        model_name = 'SEIHRVS' if vacinacao_ativa else 'SEIHRS'
        ax[0].set_title(f'Simulação epidemiológica {model_name}')
        ax[0].plot(t/365, x[0,:], 'r', label = 'S')
        ax[0].plot(t/365, x[1,:], 'g', label = 'E')
        ax[0].plot(t/365, x[2,:], 'b', label = 'I')
        ax[0].plot(t/365, x[3,:], 'm', label = 'H')
        ax[0].plot(t/365, x[4,:], 'y', label = 'R')
        if vacinacao_ativa:
            ax[0].plot(t/365, x[5,:], 'c', label = 'V')
            ax[0].axvline(tempo_inicio_vacinacao_anos, color='k', linestyle='-', linewidth=1.5, label='Início da vacinação')
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