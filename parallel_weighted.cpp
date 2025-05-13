#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <thread>

using namespace std;

const string PYTHON_ENV = "/Users/illiamatsko/My/University/paralel/HyperparameterTuning/.venv/bin/python3";
const string PYTHON_SCRIPT = "/Users/illiamatsko/My/University/paralel/HyperparameterTuning/model/lstm.py";

vector<string> global_indicators = {"SMA", "EMA", "RSI", "MACD", "BB_UP", "ATR", "BB_DOWN", "High", "Low", "Open", "Volume"};

struct Parameter
{
    double value;
    double weight;

    Parameter(const double v, const double w) : value(v), weight(w) {}
};

struct Chromosome
{
    vector<string> indicators;
    Parameter layers;
    Parameter units;
    Parameter dropout;
    Parameter epochs;
    Parameter batch_size;
    Parameter window_size;
    double fitness;

    Chromosome(const double layers_val, const double units_val, const double dropout_val, const double epochs_val, const double batch_size_val, const double window_size_val, const double fitness_val)
        : indicators(global_indicators), layers(layers_val, 0), units(units_val, 0), dropout(dropout_val, 0),
          epochs(epochs_val, 0), batch_size(batch_size_val, 0), window_size(window_size_val, 0), fitness(fitness_val)
    {
    }

    void print() const
    {
        cout << "Indicators: ";
        for (const auto &ind : indicators)
        {
            cout << ind << " ";
        }
        cout << "\nLayers: " << layers.value << "  Weight: " << layers.weight
             << "\nUnits: " << units.value << "  Weight: " << units.weight
             << "\nDropout: " << dropout.value << "  Weight: " << dropout.weight
             << "\nEpochs: " << epochs.value << "  Weight: " << epochs.weight
             << "\nBatch Size: " << batch_size.value << "  Weight: " << batch_size.weight
             << "\nWindow Size: " << window_size.value << "  Weight: " << window_size.weight
             << "\nFitness (Error): " << fitness << "%" << endl;
    }
};

class GeneticAlgorithm
{
private:
    mt19937 gen{random_device{}()};
    int population_size;
    int generations;
    double mutation_rate;
    vector<Chromosome> previous_population;
    vector<Chromosome> current_population;

    const int MIN_LAYERS = 1;
    const int MAX_LAYERS = 2;
    const int MIN_UNITS = 10;
    const int MAX_UNITS = 20;
    const double MIN_DROPOUT = 0.05;
    const double MAX_DROPOUT = 0.5;
    const int MIN_EPOCHS = 2;
    const int MAX_EPOCHS = 3;
    const int MIN_BATCH_SIZE = 16;
    const int MAX_BATCH_SIZE = 128;
    const int MIN_WINDOW_SIZE = 10;
    const int MAX_WINDOW_SIZE = 90;

public:
    GeneticAlgorithm(const int pop_size, const int gen_count, const double mut_rate)
    {
        population_size = pop_size;
        generations = gen_count;
        mutation_rate = mut_rate;

        initializePopulation();
    }

    void initializePopulation()
    {
        current_population.clear();
        current_population.reserve(population_size);

        for (int i = 0; i < population_size; ++i)
        {
            current_population.push_back(createRandomChromosome());
        }

        previous_population = current_population;
    }

    Chromosome createRandomChromosome()
    {
        uniform_int_distribution<> layers_dist(MIN_LAYERS, MAX_LAYERS);
        uniform_int_distribution<> units_dist(MIN_UNITS, MAX_UNITS);
        uniform_real_distribution<> dropout_dist(MIN_DROPOUT, MAX_DROPOUT);
        uniform_int_distribution<> epochs_dist(MIN_EPOCHS, MAX_EPOCHS);
        uniform_int_distribution<> batch_dist(MIN_BATCH_SIZE, MAX_BATCH_SIZE);
        uniform_int_distribution<> window_dist(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE);
        vector<string> indicators_val = global_indicators;
        const double layers_val = layers_dist(gen);
        const double units_val = units_dist(gen);
        const double dropout_val = dropout_dist(gen);
        const double epochs_val = epochs_dist(gen);
        const double batch_size_val = batch_dist(gen);
        const double window_size_val = window_dist(gen);
        constexpr double fitness_val = 100000;

        return {layers_val, units_val, dropout_val, epochs_val, batch_size_val, window_size_val, fitness_val};
    }

    void createFiles(vector<Chromosome> &population) const
    {
        for (int i = 0; i < population_size; i++)
        {
            Chromosome chrom = population[i];
            ofstream params_file("/Users/illiamatsko/My/University/paralel/GenAlg/params_" + to_string(i) + ".txt");
            string possible_indicators_str;
            for (const auto &ind : global_indicators)
            {
                possible_indicators_str += ind + " ";
            }
            string indicators_str;
            for (const auto &ind : chrom.indicators)
            {
                indicators_str += ind + " ";
            }

            params_file << "possible_indicators=" << possible_indicators_str << "\n"
                        << "indicators=" << indicators_str << "\n"
                        << "window_size=" << chrom.window_size.value << "\n"
                        << "epochs=" << chrom.epochs.value << "\n"
                        << "batch_size=" << chrom.batch_size.value << "\n"
                        << "layers=" << chrom.layers.value << "\n"
                        << "units=" << chrom.units.value << "\n"
                        << "dropout=" << chrom.dropout.value << "\n";
            params_file.close();
        }
    }

    static string execPython(const string &command)
    {
        string result;
        char buffer[128];

        FILE *pipe = popen(command.c_str(), "r");
        if (!pipe)
            throw runtime_error("popen() failed!");

        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        {
            result += buffer;
        }

        if (const int rc = pclose(pipe); rc != 0)
        {
            cerr << "Python script returned non-zero exit code: " << rc << endl;
            throw runtime_error("Python script execution failed!");
        }

        return result;
    }

    void evaluatePopulation(vector<Chromosome> &population, const int num_threads) const
    {
        createFiles(population);

#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int i = 0; i < population_size; i++)
        {
            string command = PYTHON_ENV + ' ' + PYTHON_SCRIPT + " /Users/illiamatsko/My/University/paralel/GenAlg/params_" + to_string(i) + ".txt";
            string output = execPython(command);

            population[i].fitness = stod(output);
            cout << "Chromosome " << i << "/" << population_size - 1 << ". Fitness: " << output << '%' << endl;
        }
    }

    void updateWeights(const int pos, const double learning_rate)
    {
        const double fitness_current = current_population[pos].fitness;
        const double fitness_previous = previous_population[pos].fitness;
        const double fitness_change = (fitness_previous - fitness_current) / abs(fitness_previous);

        const double dropout_current = current_population[pos].dropout.value;
        const double dropout_previous = previous_population[pos].dropout.value;
        double param_change = abs(dropout_previous - dropout_current) / abs(dropout_previous);

        double influence = fitness_change * param_change;

        const double updated_weight = tanh(influence * learning_rate);
        current_population[pos].dropout.weight += updated_weight;

        const double layers_current = current_population[pos].layers.value;
        const double layers_previous = previous_population[pos].layers.value;
        param_change = abs(layers_previous - layers_current) / abs(layers_previous);
        influence = fitness_change * param_change;
        current_population[pos].layers.weight += tanh(influence * learning_rate);

        const double units_current = current_population[pos].units.value;
        const double units_previous = previous_population[pos].units.value;
        param_change = abs(units_previous - units_current) / abs(units_previous);
        influence = fitness_change * param_change;
        current_population[pos].units.weight += tanh(influence * learning_rate);

        const double epochs_current = current_population[pos].epochs.value;
        const double epochs_previous = previous_population[pos].epochs.value;
        param_change = abs(epochs_previous - epochs_current) / abs(epochs_previous);
        influence = fitness_change * param_change;
        current_population[pos].epochs.weight += tanh(influence * learning_rate);

        const double batch_size_current = current_population[pos].batch_size.value;
        const double batch_size_previous = previous_population[pos].batch_size.value;
        param_change = abs(batch_size_previous - batch_size_current) / abs(batch_size_previous);
        influence = fitness_change * param_change;
        current_population[pos].batch_size.weight += tanh(influence * learning_rate);

        const double window_size_current = current_population[pos].window_size.value;
        const double window_size_previous = previous_population[pos].window_size.value;
        param_change = abs(window_size_previous - window_size_current) / abs(window_size_previous);
        influence = fitness_change * param_change;
        current_population[pos].window_size.weight += tanh(influence * learning_rate);
    }

    Chromosome mutate(Chromosome chrom)
    {
        normal_distribution<> rand_norm(0.0, 1.0);

        auto mutate_param = [&](Parameter &param, const double min_val, const double max_val, const bool is_integer = false)
        {
            const double weight = tanh(param.weight);
            const double influence = 1.0 - abs(weight);

            double delta = rand_norm(gen) * influence * (max_val - min_val) * 0.1;

            param.value += delta;

            param.value = clamp(param.value, min_val, max_val);

            if (is_integer)
            {
                param.value = round(param.value);
            }
        };

        mutate_param(chrom.layers, MIN_LAYERS, MAX_LAYERS, true);
        mutate_param(chrom.units, MIN_UNITS, MAX_UNITS, true);
        mutate_param(chrom.dropout, MIN_DROPOUT, MAX_DROPOUT, false);
        mutate_param(chrom.epochs, MIN_EPOCHS, MAX_EPOCHS, true);
        mutate_param(chrom.batch_size, MIN_BATCH_SIZE, MAX_BATCH_SIZE, true);
        mutate_param(chrom.window_size, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, true);

        return chrom;
    }

    void run(const double learning_rate, const int num_threads)
    {
        cout << "=== Starting Parallel Weighted Genetic Algorithm ===" << endl;
        cout << "Population size: " << population_size << ", Generations: " << generations << endl;
        cout << "Number of threads: " << num_threads << "\n\n";

        Chromosome best_overall = Chromosome(0, 0, 0, 0, 0, 0, 100000.0);

        cout << "Evaluating first initial population..." << endl;
        evaluatePopulation(previous_population, num_threads);

        for (int i = 0; i < population_size; i++)
        {
            if (previous_population[i].fitness < best_overall.fitness)
            {
                best_overall = previous_population[i];
            }
        }

        cout << "Best chromosome after first initial evaluation:" << endl;
        best_overall.print();

        for (int i = 0; i < population_size; i++)
        {

            current_population[i] = createRandomChromosome();
        }

        cout << "\nEvaluating second initial population..." << endl;
        evaluatePopulation(current_population, num_threads);

        for (int i = 0; i < population_size; i++)
        {
            if (current_population[i].fitness < best_overall.fitness)
            {
                best_overall = current_population[i];
            }
        }

        cout << "Best chromosome after second initial evaluation:" << endl;
        best_overall.print();

        for (int i = 0; i < population_size; i++)
        {

            updateWeights(i, learning_rate);
        }

        previous_population = current_population;

        for (int gen = 0; gen < generations; gen++)
        {
            cout << "\n--- Generation " << gen + 1 << " ---" << endl;

            for (int pos = 0; pos < population_size; pos++)
            {
                current_population[pos] = mutate(previous_population[pos]);
            }

            evaluatePopulation(current_population, num_threads);

            for (int i = 0; i < population_size; i++)
            {

                updateWeights(i, learning_rate);
            }

            for (int i = 0; i < population_size; i++)
            {
                if (current_population[i].fitness < best_overall.fitness)
                {
                    best_overall = current_population[i];
                }
            }
            cout << "Best chromosome after generation " << gen + 1 << ":" << endl;
            best_overall.print();

            previous_population = current_population;
        }

        cout << "\n\n=== Best chromosome after all generations ===" << endl;
        best_overall.print();
    }
};

int main()
{
    constexpr int population_size = 8;
    constexpr int generations = 15;
    constexpr double mutation_rate = 0.2;

    constexpr double learning_rate = 2;
    constexpr int num_threads[] = {2, 4, 8, 16};

    GeneticAlgorithm genalg = GeneticAlgorithm(population_size, generations, mutation_rate);

    for (const int threads : num_threads)
    {
        const auto start_time = chrono::high_resolution_clock::now();

        genalg.run(learning_rate, threads);

        const auto end_time = chrono::high_resolution_clock::now();
        cout << "Execution time: " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << " milliseconds" << "\n\n\n\n";
    }

    return 0;
}