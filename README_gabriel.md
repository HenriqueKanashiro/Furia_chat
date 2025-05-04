# PoC Image Classification

Autores:

- Nome (email)

## **Aviso**

A leitura desta documentação e o entendimento da arquitetura deste projeto serão mais fáceis e melhor aproveitadas se você estiver familiarizado com os seguintes serviços do [Google Cloud Platform (GCP)](https://cloud.google.com/): [Google Cloud Storage](https://cloud.google.com/storage/docs), [Vertex AI Workbench Instances](https://cloud.google.com/vertex-ai/docs/workbench/instances/introduction) e, especialmente,[Vertex AI Auto ML](https://cloud.google.com/automl?hl=en) e seus termos técnicos.

## 1. Contexto

A Companhia Paranaense de Energia (Copel) realiza diariamente cerca de 180 serviços de roçagem em áreas rurais, sob redes de distribuição de energia elétrica. A comprovação da completude desses serviços é feita por amostragem, com análise manual de fotos enviadas por prestadores — processo que demanda tempo de colaboradores e pode gerar custos operacionais elevados.
Diante desse cenário, esta Prova de Conceito (PoC) visa desenvolver um modelo de classificação de imagens baseado em inteligência artificial, utilizando recursos do Vertex AI AutoML, com o objetivo de automatizar a verificação da completude dos serviços.

A abordagem consiste em utilizar um modelo de visão computacional supervisionado para classificar as imagens em duas categorias (multiclasse):

- Roçagem aprovada;
- Roçagem não aprovada;

O projeto adota como premissa a utilização de imagens previamente rotuladas por analistas da Copel, permitindo o uso da metodologia de aprendizado supervisionado.

### 1.1 Objetivo

Desenvolver uma PoC com base em modelos AutoML para classificar imagens relacionadas aos serviços de roçagem, de forma automatizada, com base em dados já rotulados e armazenados na nuvem.
O principal objetivo é validar se é possível criar um modelo performático para realizar essa classificação utilizando os dados já existentes da Copel. 

### 1.2 Criterios de sucesso

- Prova de Conceito realizada, com testes e analises reportadas para identificar a viabilidade do projeto.
- Código, documentação, análises e reports entregues ao cliente.

## 2. Arquitetura

A arquitetura proposta para esta PoC foi desenvolvida sobre os serviços do Google Cloud Platform, pensando em ser uma arquitetura rapida e que facilite o treinamento e implementação de modelos de Machine Learning, foi escolhido como principal serviço para o treinamento dos modelos, o Vertex AI AutoML junto com o Vertex AI Workbench.

Abaixo, a visão geral da arquitetura:

||
|:-:|
|<img src="images/arquitetura.png" width=60%>|

A arquitetura final da solução pode ser descrita da seguinte forma:

1. A partir de uma instância do Vertex AI Workbench, um kernel do JupyterLab é criado.
2. Via notebook JupyterLab, as imagens utilizadas para a criação dos datasets gerenciados são obtidas no Cloud Storage.
3. As imagens são, então, utilizadas para treino, teste e validação de um modelo classificador no Vertex AI, utilizando AutoML.

### 2.1 Etapas do Processo

1. Armazenamento de imagens:
Imagens rotuladas são armazenadas em buckets no Google Cloud Storage, separadas por prefixos/pastas conforme suas classes.

2. Execução via notebook (Vertex AI Workbench):  

Um kernel Jupyter executa scripts Python que:

- Capturam as imagens do GCS;
- Geram o CSV de referência para o AutoML;
- Criam datasets e disparam o treinamento de modelos.

3. Treinamento com AutoML (Vertex AI):

O modelo é treinado automaticamente com base nos dados enviados, respeitando a divisão entre treino, validação e teste.

4. Avaliação automática do modelo:

Métricas como auPRC, precision, recall, logLoss e matriz de confusão são obtidas via API do Vertex AI.

5. Deploy do modelo:

Após treinado, o modelo é publicado em um endpoint gerenciado, ficando disponível para inferência via API REST.

## 3. Serviços Utilizados

### 3.1 Google Cloud Storage

O Google Cloud Storage (GCS) é utilizado como repositório central para os dados de entrada e intermediários da solução:

- As imagens rotuladas são armazenadas em buckets organizados por prefixo (ex: `1-Aprovadas/` e `2-Reprovadas/`), representando as classes para treinamento.
- O arquivo `.csv` de configuração do dataset (gerado automaticamente pelo código) também é armazenado no GCS, em um diretório chamado `ConfigDataset/`.

O AutoML acessa diretamente este CSV como entrada para a criação do dataset.

```
gs://bucket_copel/
├── 1-Aprovadas/
│   ├── img001.jpg
│   └── ...
├── 2-Reprovadas/
│   ├── img002.jpg
│   └── ...
└── ConfigDataset/
    └── dataset.csv
```

### 3.2 Vertex AI Workbench

O **Vertex AI Workbench** é o ambiente computacional onde toda a lógica do projeto é implementada e executada. Utilizamos uma instância JupyterLab com permissões GCP para acessar os recursos diretamente via SDK.

Neste ambiente são realizadas:

- Execução do pipeline completo: preparação dos dados, treinamento, avaliação e deploy do modelo;
- Importação das bibliotecas e chamadas à API do Vertex AI;
- Visualizações gráficas de métricas do modelo e matriz de confusão.

A instância é criada com acesso à **conta de serviço** que possui permissões para manipular recursos do Vertex AI e do GCS.

### 3.3 Vertex AI Datasets

O recurso Vertex AI Datasets é utilizado para armazenar e gerenciar os dados de entrada estruturados para o treinamento do modelo AutoML. Após a geração e upload do arquivo `.csv` contendo os paths das imagens e seus respectivos rótulos, este arquivo é utilizado para importar os dados e criar um dataset no Vertex AI.

Funcionamento

- O dataset é criado programaticamente via código, com base no CSV armazenado no GCS;
- O esquema de importação depende se o problema é single-label ou multi-label:
  - `image_classification_single_label_io_format_1.0.0.yaml`
  - `image_classification_multi_label_io_format_1.0.0.yaml`

O arquivo `.csv` que será utilizado para geração do dataset, precisa seguir um **schema** predefinido pelo Google:

<img src="images/csv_format.png" width=100%>

Após a criação, os dados são automaticamente divididos nos conjuntos de treino, validação e teste com base nas proporções informadas ou nos filtros definidos (training_filter_split, validation_filter_split, etc.).

A manipulação e reuso de datasets gerenciados evita a necessidade de criar múltiplas versões idênticas, otimizando o uso de recursos e simplificando testes A/B.

### 3.4 Vertex AI AutoML

O Vertex AI AutoML é o componente responsável pelo treinamento dos modelos de classificação de imagem. Os principais pontos incluem:

- Suporte para classificação de datasets tipo **single-label** e **multi-label**;
- Criação de datasets via CSV contendo caminhos GCS e classes associadas;
- Divisão de dados em treino, validação e teste por percentual ou por filtros personalizados (training_filter_split, etc);
- Suporte a early stopping, budget controlado por node-hour, e geração automática de métricas.

O modelo é treinado com base nos dados fornecidos, sem a necessidade de definir manualmente arquiteturas ou parâmetros — o AutoML cuida disso automaticamente. Porém é possivel escolher alguns tipos de modelos para serem treinados, como no caso de classificação de imagens:

* `CLOUD`: Modelo custo-benefício.
* `CLOUD_1`: Modelo mais performático, porém com maior treinamento.

Para saber mais sobre os possíveis tipo de modelos disponíveis na Google Cloud, acesse essa [documentação oficial](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.AutoMLImageTrainingJob).

Para fazer requisições em um modelo do AutoML, existem 2 possibilidades, sendo elas, previsão em lote e previsão online.

* Previsão em lote: Esse tipo de previsão é feito através de uma chamada **assíncrona** diretamente no modelo que está hospedado no **Model Registry**. Para a previsão em lote, não é necessário que o modelo esteja implementado atrás de um endpoint.
* Previsão Online: Esse tipo de previsão é feito através de uma chamada **síncrona**, e necessita que exista um endpoint gerenciado do Vertex AI, para que o modelo possa ser alcançado  via HTTP Rest.

### 3.5 Vertex AI Endpoints

Após o treinamento, os modelos são disponibilizados para inferência por meio do serviço Vertex AI Endpoints:

- Criação de um endpoint gerenciado para servir o modelo;
- Deploy do modelo para o endpoint, com configuração de réplicas, controle de tráfego e metadados;
- A inferência pode ser feita via chamadas REST para o endpoint, enviando imagens como payload.

Esse componente é essencial para validar a PoC em um cenário mais próximo do ambiente de produção.

O endpoint escolhido para implementar o modelo do AutoML treinado durante o projeto, foi o **Shared public endpoint**. Essa escolha foi feita pois ele é o único endpoint suportado pelo AutoML. Para mais informações sobre como funcionam os endpoints acesse o link da [documentação oficial](https://cloud.google.com/vertex-ai/docs/predictions/choose-endpoint-type).

## 4. Código do Projeto

O código-fonte deste projeto foi pensado para que seja fácil de reproduzir os experimentos realizados com o AutoML, por isso ele está organizado em três classes principais: Classifier, ModelUtils e Deployer. Juntas, elas encapsulam o ciclo completo de criação de datasets, treinamento, avaliação e deploy de modelos no Vertex AI AutoML. As classes foram projetadas para serem utilizadas diretamente em notebooks hospedados no Vertex AI Workbench.

### 4.1 Classifier: criação do dataset e treinamento do modelo

A classe Classifier realiza o pipeline de preparação de dados e execução do treinamento no AutoML. Suas principais funcionalidades incluem:

- Leitura das imagens no Cloud Storage a partir de dois prefixos `1-Aprovadas/` e `2-Reprovadas/`, que serão utilizadas para a criação do dataset;
- Geração de um CSV no formato esperado pelo AutoML e upload para o bucket GCS na pasta `ConfigDataset/`;
- Criação de um dataset gerenciado no Vertex AI, a partir do arquivo `.csv` e das imagens;
- Execução do job de treinamento (com suporte para classificação single-label ou multi-label).

Exemplo de uso:

```python
classifier = Classifier(
    project="id_projeto", # Id do projeto
    general_display_name="nome_modelo", 
    bucket_name="nome_bucket", # Nome do bucket
    approved_prefix="1-Aprovadas/", # Nome da pasta onde estão as imagens aprovadas
    rejected_prefix="2-Reprovadas/", # Nome da pasta onde estão as imagens reprovadas
    multi_label=False, # Se o dataset e o modelo serão multi_label ou não.

    # Filtros para controle manual dos dados de treinamento, teste e validação
    training_filter_split="labels.aiplatform.googleapis.com/ml_use=training",
    validation_filter_split="labels.aiplatform.googleapis.com/ml_use=validation",
    test_filter_split="labels.aiplatform.googleapis.com/ml_use=test"
)
# A função `create_automl_dataset()` pode receber opcionalmente o parâmetro `gcs_output_path` com o caminho para o arquivo `.csv` caso ele tenha sido gerado manualmente.
classifier.create_automl_dataset() # cria o dataset 
classifier.create_training_job() # cria o job de treinamento
```

### 4.2  ModelUtils: avaliação e visualização de métricas

A classe ModelUtils fornece ferramentas para inspeção dos modelos treinados:

- Listagem de modelos e suas avaliações;
- Extração de métricas como precision, recall, logLoss, auPrc;
- Visualização gráfica (Precision vs Recall vs Threshold);
- Plotagem da matriz de confusão em porcentagem por classe.

Exemplo de uso:

```python
utils = ModelUtils(
    project_number="numero_projeto",
    location="us-central1",
    model_id="id_modelo"
)

metrics = utils.get_model_evaluation() # Obtém as métricas do modelo
utils.write_metrics(metrics) # Formata as métricas obtidas
utils.plot_graphic(metrics) # Constrói um gráfico a partir das métricas
utils.plot_confusion_matrix(metrics) # Constrói a matriz de confusão 
```

Para mais informações sobre como extrair e analisar as métricas de treinamento dos modelos do AutoML, consulte essa [documentação oficial](https://cloud.google.com/vertex-ai/docs/image-data/classification/evaluate-model).

### 4.3 Deployer: publicação de modelos

A classe Deployer permite realizar o deploy do modelo treinado para um endpoint no Vertex AI, viabilizando a inferência via API REST, com as seguintes funcionalidades:

- Criação de endpoints gerenciados;
- Deploy de modelos com controle de replicação e split de tráfego;
- Gerenciamento de endpoints existentes.

```python
deployer = tools.Deployer(
    project="PROJECT_ID",
    location="REGION",
    model_id="model_id",
    deployed_model_display_name="name"
)

deployer.create_endpoint() # Cria um endpoint no Vertex AI
deployer.deploy_model() # Implementa um modelo em um endpoint do Vertex AI
```

## 5. Experimentos 

Models Ids:

- single_label_balanced_cloud = 9173653220558372864
- single_label_unbalanced_cloud = 4635713646029176832
- single_label_balanced_cloud_1 = 3663499076470571008
- multi_label_balanced_cloud = 4909166585903579136
- multi_label_unbalanced_cloud = 58367574760488960
- multi_label_balanced_cloud_1 = 2692973356772229120

| Metrics | balanced_dataset | unbalanced_dataset | balanced_dataset_v2 |
|-------------|:-------------:|:-------------:|:-------------:|
| Total images | 830 | 2,236 | 1,244 |
| Training images | 664 | 1,789 | 996 |
| Validation images | 83 | 224 | 125 |
| Test images | 83 | 223 | 123 |

| Metrics | single_label_balanced_cloud | single_label_unbalanced_cloud | single_label_balanced_cloud_1 
|-------------|:-------------:|:-------------:|:-------------:|
| Confidence threshold | 0.5 | 0.5 | 0.5 |
| Average precision | 0.85612017 | 0.94126695 | 0.82053 |
| Precision | 79,5% | 88,3% | 80,7% |
| Recall | 79,5% | 88,3% | 80,7% |
| logLoss | 1.0158305 | 0.94561625 | 0.5806413 |
| Confusion Matrix | ![single_label_balanced_cloud](images/cm_single_label_balanced_cloud.png) | ![single_label_unbalanced_cloud](images/cm_single_label_unbalanced_cloud.png) | ![single_label_balanced_cloud](images/cm_single_label_balanced_cloud_1.png) |

| Metrics | multi_label_balanced_cloud | multi_label_unbalanced_cloud | multi_label_balanced_cloud_1
|-------------|:-------------:|:-------------:|:-------------:|
| Confidence threshold | 0.5 | 0.5 | 0.5 |
| Average precision | 0.91560507 | 0.9509373 | 0.85653514 |
| Precision | 85.5% | 91.9% | 77% |
| Recall | 85.5% | 81.3% | 70.2% |
| logLoss | 1.7191641 | 0.33811066 | 0.47374645 |
| Confusion Matrix | ![multi_label_balanced_cloud](images/cm_multi_label_balanced_cloud.png) | ![multi_label_unbalanced_cloud](images/cm_multi_label_unbalanced_cloud.png) | ![multi_label_balanced_cloud_1](images/cm_multi_label_balanced_cloud_1.png) |

## 6. Próximos passos

* Implementação do ciclo de vida completo de Machine Learning;
* Ter um dataset de treinamento maior e balanceado, seguindo as boas práticas de pelo menos 1000 imagens por rótulo;
