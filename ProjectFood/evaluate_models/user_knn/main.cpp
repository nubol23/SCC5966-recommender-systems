#include <iostream>
#include <rapidcsv.h>

#define ARMA_NO_DEBUG

#include <armadillo>
#include <indicators.hpp>
#include <algorithm>

using namespace std;
using namespace arma;
using namespace indicators;

auto create_bar(const int n_iters) {
    return indicators::ProgressBar(
            option::BarWidth{50},
            option::Start{" ["},
            option::Fill{"="},
            option::Lead{">"},
            option::Remainder{"."},
            option::End{"]"},
            option::ForegroundColor{Color::white},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::MaxProgress{n_iters},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    );
}

sp_mat build_ratings_matrix(const rapidcsv::Document &csv, const int n_users, const int n_items) {
    auto interactions = csv.GetColumn<uword>("user_id");
    auto n_interactions = interactions.size();

    auto tmp_items = csv.GetColumn<uword>("movie_id");
    interactions.insert(interactions.end(), tmp_items.begin(), tmp_items.end());

    umat locations(interactions);
    locations.reshape(n_interactions, 2);
    locations = locations.t();

    auto values = vec(csv.GetColumn<double>("rating"));

    sp_mat ratings_mat(locations, values, n_users, n_items);

    return ratings_mat;
}

vec get_means(const sp_mat &ratings, double eps=1e-15) {
    size_t n_users = ratings.n_rows;
    vec acc(n_users, fill::zeros);
    vec counts(n_users, fill::zeros);

    auto bar = create_bar((int) n_users);
    for (int u = 0; u < n_users; u++) {
        for (auto elem: ratings.row(u)) {
            acc(u) += elem;
            counts(u) += 1;
        }
        bar.tick();
    }

    return acc/(counts+eps);
}

double sim(const uword u, const uword v, const sp_mat &ratings, const vec &means, double eps=1e-15) {
    vector<uword> items_idxs;

    // Get items rated by both
    map<uword , int> items_rated_count;
    // items rated by user u
    auto row_u = ratings.row(u);
    auto start = row_u.begin();
    auto end = row_u.end();
    for (auto it = start; it != end; ++it)
        items_rated_count[it.col()]++;

    auto row_v = ratings.row(v);
    start = row_v.begin();
    end = row_v.end();
    for (auto it = start; it != end; ++it)
        items_rated_count[it.col()]++;

    for (auto x: items_rated_count)
        if (x.second == 2)
            items_idxs.push_back(x.first);

    // Calculate similarity
    double n_ui = 0, n_vi = 0, cov_ui_vi = 0;
    double r_ui_r, r_vi_r;
    for (uword i: items_idxs) {
        r_ui_r = ratings(u, i) - means(u);
        r_vi_r = ratings(v, i) - means(v);

        cov_ui_vi += (r_ui_r * r_vi_r);
        n_ui += r_ui_r*r_ui_r;
        n_vi += r_vi_r*r_vi_r;
    }

    return cov_ui_vi/(sqrt(n_ui) * sqrt(n_vi) + eps);
}

double pred(const uword u, const uword i, const int k, const sp_mat &ratings, const vec &means, double eps=1e-15) {
    vector<tuple<double, double, double>> similarities;

    auto col_i = ratings.col(i);
    auto start = col_i.begin();
    auto end = col_i.end();
    for (auto it = start; it != end; ++it) {
        uword v = it.row();
        if (u != v) {
            double sim_uv = sim(u, v, ratings, means);
            similarities.emplace_back(sim_uv, ratings(v, i), means(v));
        }
    }

    double numerator = 0, denominator = 0;
    sort(similarities.begin(), similarities.end(), greater<>());
    for (int f = 0; f < k; f++) {
        auto &[similarity_uv, rating_vi, mean_v] = similarities[f];
        numerator += similarity_uv*(rating_vi - mean_v);
        denominator += similarity_uv;
    }

    return means(u) + numerator/(denominator + eps);
}

double clip(const double val, const double low, const double high) {
    if (val < low)
        return low;
    if (high < val)
        return high;

    return val;
}

vec predict_batch(const vector<int> &users, const vector<int> &items, const sp_mat &ratings, const int neighbours) {
    vec means = get_means(ratings);

    vec predictions(users.size(), fill::zeros);

    auto bar = create_bar((int) users.size());
    for (int idx = 0; idx < users.size(); idx++) {
        uword user_id = users[idx];
        uword item_id = items[idx];
        double prediction = clip(pred(user_id, item_id, neighbours, ratings, means), 1, 5);

        predictions(idx) = prediction;

        bar.tick();
    }

    return predictions;
}

int main() {
    int users = 3974;
    int movies = 3564;

//    string base_path = "/home/nubol23/Desktop/Codes/USP/SCC5966/kaggle/notebooks/ProjectFood/Preprocessing/processed_dataframes/";
    string base_path = "/home/nubol23/Desktop/Codes/USP/SCC5966/kaggle/notebooks/ProjectFood/evaluate_models/user_knn";
    rapidcsv::Document train_csv(base_path + "/train.csv");
    rapidcsv::Document val_csv(base_path + "/val.csv");
//    rapidcsv::Document test_csv(base_path + "/test.csv");

    sp_mat ratings_mat = build_ratings_matrix(train_csv, users, movies);

    for (int k = 0; k < 1; k++) {
        vec predictions = predict_batch(val_csv.GetColumn<int>("user_id"),
                                        val_csv.GetColumn<int>("movie_id"),
                                        ratings_mat,
                                        k);
        cout<<predictions<<endl;
    }

  return 0;
}