import yaml
import pandas as pd

def main():
    with open('output_o1_mini.yaml', 'r') as file:
        output = yaml.safe_load(file)

    elements = []
    for conversation in output['conversaciones']:
        string = 'preguntas:\n'
        for question in conversation['preguntas']:
            string += f'- pregunta: "{question["pregunta"]}"\n'
            if question['contexto']:
                string += f'  contexto: "{question["contexto"]}"\n'
        elements.append(string)

    data = pd.DataFrame({'conversacion': elements})
    data.to_csv('output_o1_mini.csv', index=False)

if __name__ == "__main__":
    main()