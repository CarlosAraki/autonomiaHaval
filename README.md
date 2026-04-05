# Custo de abastecimento em veículo híbrido — modelo analítico e simulação

Estudo em Python para **comparar o custo acumulado de energia** (gasolina e bateria) ao longo da **vida útil de 500.000 km**, em função da **proporção de uso em cidade versus estrada** e de **três estratégias de uso** do híbrido. O código gera curvas comparativas (2D), superfícies 3D com marcos de quilometragem e um resumo tabular via `pandas`.

---

## Objetivo

Quantificar e visualizar **quanto se gasta em R$** para percorrer 500.000 km com viagens de **20 km**, quando a fração de quilometragem em **cidade** varia de **1% a 100%** (passo de 1 ponto percentual), contrastando:

1. um **baseline** de consumo cíclico fixo;
2. um **comportamento estocástico** (prioridade elétrica com esquecimento e regras de abastecimento);
3. um **uso otimizado** por tipo de vía.

---

## Premissas principais

| Parâmetro | Valor |
|-----------|--------|
| Vida útil | 500.000 km |
| Comprimento médio de viagem | 20 km (25.000 viagens no total) |
| Eixo de análise | Proporção de km em **cidade** *p* (1% a 100%); estrada = 1 − *p* |
| Autonomia (capacidade) | Gasolina: **800 km**; elétrico: **200 km** (modelados em segmentos de 20 km) |

### Rendimento energético (km por R$)

O custo marginal em **R$/km** é o inverso do rendimento:

| Fonte | Cidade | Estrada |
|-------|--------|---------|
| Gasolina | 2 km/R$ → **0,50** R$/km | 2,5 km/R$ → **0,40** R$/km |
| Bateria | 5 km/R$ → **0,20** R$/km | 3 km/R$ → **≈0,333** R$/km |

**Abastecimento e recarga** no modelo **não** incluem taxa fixa por “tanque cheio” ou “recarga completa”: apenas **recompõem autonomia**. O custo monetário entra **por quilômetro percorrido**, conforme o modo (gasolina ou elétrico) e o **par cidade/estrada** atribuído a cada viagem (ver Caso 2).

---

## Os três cenários

### Caso 1 — Baseline (ciclo fixo)

A cada **1.000 km** o padrão é **800 km em gasolina** e **200 km em elétrico**. Esses blocos são precificados como se a **mesma** proporção \(p\) de cidade/estrada do cenário se aplicasse a todos os km do ciclo:

- custo dos 800 km de gasolina: média ponderada entre R$/km cidade e estrada com pesos *p* e 1 − *p*;
- idem para os 200 km elétricos.

O custo total em 500.000 km é **500 × (custo de um ciclo de 1.000 km)**. Varia com a proporção *p*, ao contrário de um baseline puramente constante.

### Caso 2 — Prioridade elétrica com comportamento probabilístico

Simulação **Monte Carlo vetorizada** (`numpy`): muitas trajetórias em paralelo, um passo por viagem de 20 km.

- Prioriza elétrico enquanto houver autonomia e o estado lógico indicar modo “elétrico”.
- Com **1 segmento elétrico restante** (20 km na bateria): **95%** lembra de recarregar (autonomia elétrica restaurada; custo só dos km rodados); **5%** esquece e a viagem tende a sair na **gasolina** (se houver combustível / regras de tanque vazio).
- Em viagem na **gasolina**: após consumir o segmento, **5%** lembra de recarregar a bateria e volta ao modo elétrico.
- **Tanque vazio**: **95%** enche o tanque (restaura segmentos de gasolina); **5%** “esquece” — se ainda houver bateria, pode usar elétrico nessa viagem; caso contrário, abastece de forma obrigatória.

Para cada trajetória, o código acumula dois totais hipotéticos por viagem:

- **soma se todas as viagens fossem em cidade**;
- **soma se todas fossem em estrada**.

Assumindo que, para uma proporção *p* de km em cidade, cada viagem é **cidade** com probabilidade *p* (i.i.d.), o **custo esperado** é **linear em *p***:

**E[custo | *p*] = *p* × S_cidade + (1 − *p*) × S_estrada.**

Uma única rodada de simulação permite plotar a curva do Caso 2 para **todos** os valores de *p* no grid. O desvio padrão reportado no console refere-se a uma referência **50% cidade / 50% estrada** sobre as mesmas trajetórias.

### Caso 3 — Chaveamento inteligente

Uso idealizado: **somente elétrico na cidade** e **somente gasolina na estrada**. O custo em 500.000 km é **500.000 × (*p* × c_el,cidade + (1 − *p*) × c_gas,estrada)**, com *c* em R$/km conforme a tabela de rendimento.

---

## Saídas geradas

| Artefato | Descrição |
|----------|-----------|
| `custo_abastecimento_hibrido.png` | Comparação 2D: custo total × proporção de uso na cidade (áreas + linhas). |
| `custo_abastecimento_hibrido_3d.png` | Três superfícies: % cidade × km acumulado (de 5.000 em 5.000 até 500.000) × custo acumulado (R$). |
| Console | Mínimo/máximo por caso no horizonte de 500.000 km; desvio padrão do Caso 2 (referência 50% cidade); amostra do `DataFrame`. |

Os arquivos PNG são gravados no **mesmo diretório** do script (`Path(__file__).parent`).

---

## Requisitos e ambiente

- **Python** 3.10+ recomendado (testado com 3.13).
- Dependências: ver `requirements.txt` (`pandas`, `numpy`, `matplotlib`, `seaborn`).

### Instalação com ambiente virtual

```bash
cd autonomiaCarro
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

### Execução

```bash
python otimizacao_abastecimento_hibrido.py
```

Ou, sem ativar o venv:

```bash
./venv/bin/python otimizacao_abastecimento_hibrido.py
```

---

## Estrutura do repositório

```
autonomiaCarro/
├── README.md
├── requirements.txt
├── otimizacao_abastecimento_hibrido.py   # modelo, simulação e gráficos
├── custo_abastecimento_hibrido.png       # gerado ao rodar o script
├── custo_abastecimento_hibrido_3d.png
└── venv/                                 # opcional; não versionar
```

---

## Limitações e boas práticas de interpretação

- Os valores são **ilustrativos**: rendimentos fixos, sem inflação, manutenção, depreciação ou variação de preço de energia ao longo dos anos.
- O Caso 2 depende de **seed** (`numpy.random.Generator`) e do número de réplicas; para decisões robustas, avalie sensibilidade a `seed` e a `n_simulacoes`.
- A hipótese **i.i.d. cidade/estrada** por viagem no Caso 2 simplifica a realidade (há autocorrelação espacial/temporal no uso real).
- O Caso 3 é um **limite superior** de eficiência de roteamento por tipo de vía, não um modelo de powertrain físico.

---

## Licença e autoria

Projeto de análise quantitativa para estudo de custos. Ajuste licença e créditos conforme o uso institucional ou comercial.
